"""Index PARC-MS clips and build an offline two-hand support review gallery.

No Isaac Sim required. Read-only input; no automatic retargeting or training labels.
Rig constants/FK convention follow hhtools/io/parc_ms_skeleton.py (Apache-2.0)
and PARC data/assets/humanoid.xml. Quaternions are xyzw; world Z is up, metres.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 2
NAMES = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
         'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin',
         'right_foot', 'left_thigh', 'left_shin', 'left_foot']
PARENTS = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]
OFFSETS = np.array([[0, 0, 0], [0, 0, .236151], [0, 0, .223894], [-.02405, -.18311, .24350],
                    [0, -.274788, 0], [0, -.258947, 0], [-.02405, .18311, .24350],
                    [0, .274788, 0], [0, .258947, 0], [0, -.084887, 0], [0, 0, -.421546],
                    [0, 0, -.409870], [0, .084887, 0], [0, 0, -.421546], [0, 0, -.409870]])
LABELS = {'two_hand_crossing_candidate': '双手支撑越障候选',
          'platform_ascent_candidate': '撑上平台候选',
          'late_hand_support': '先腾空／后撑手',
          'two_hand_support_uncertain': '双手支撑，时序待核对',
          'no_sustained_two_hand_support': '无持续双手接触',
          'missing_contacts': '缺少接触标记', 'error': '读取或格式异常'}


class ArrayUnpickler(pickle.Unpickler):
    """Restrict external pickles to NumPy arrays/scalars and built-in containers."""
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'), ('numpy', 'dtype')}
        allowed |= {(m, n) for m in ('numpy.core.multiarray', 'numpy._core.multiarray')
                    for n in ('_reconstruct', 'scalar')}
        allowed |= {(m, '_frombuffer') for m in ('numpy.core.numeric', 'numpy._core.numeric')}
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported pickle global {module}.{name}')
        return super().find_class(module, name)


def decode(raw):
    return ArrayUnpickler(io.BytesIO(raw)).load() if isinstance(raw, bytes) else raw


def load_clip(path):
    outer = decode(path.read_bytes())
    m, terrain = decode(outer['motion_data']), decode(outer.get('terrain_data'))
    fps = float(m['fps'])
    root = np.asarray(m['root_pos'])
    t = len(root)
    if not np.isfinite(fps) or fps <= 0 or t < 2:
        raise ValueError('Invalid fps or fewer than two frames')
    for key, shape in [('root_pos', (t, 3)), ('root_rot', (t, 4)), ('joint_rot', (t, 14, 4))]:
        arr = np.asarray(m[key])
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f'{key}: expected finite {shape}, got {arr.shape}')
    if terrain is not None:
        hf = np.asarray(terrain['hf'])
        if hf.ndim != 2 or min(hf.shape) < 2 or not np.isfinite(hf).all() or float(terrain['dx']) <= 0:
            raise ValueError('Invalid heightfield')
        if np.asarray(terrain['min_point']).shape != (2,) or not np.isfinite(terrain['min_point']).all():
            raise ValueError('Invalid terrain origin')
    return m, terrain


def fk(m):
    positions = np.empty((len(m['root_pos']), 15, 3))
    positions[:, 0] = m['root_pos']
    rotations = [Rotation.from_quat(m['root_rot'])]
    for j in range(1, 15):
        p = PARENTS[j]
        positions[:, j] = positions[:, p] + rotations[p].apply(OFFSETS[j])
        rotations.append(rotations[p] * Rotation.from_quat(m['joint_rot'][:, j - 1]))
    return positions


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def heights(terrain, xy):
    """Sample the actual two-triangle cell split, not a bilinear approximation."""
    xy = np.asarray(xy)
    if terrain is None:
        return np.full(xy.shape[:-1], np.nan)
    hf = np.asarray(terrain['hf'])
    uv = (xy - terrain['min_point']) / float(terrain['dx'])
    valid = (uv >= 0).all(-1) & (uv <= np.array(hf.shape) - 1).all(-1)
    ij = np.floor(uv).astype(int)
    ij = np.clip(ij, 0, np.array(hf.shape) - 2)
    u, v = (uv - ij)[..., 0], (uv - ij)[..., 1]
    i, j = ij[..., 0], ij[..., 1]
    a, b, c, d = hf[i, j], hf[i + 1, j], hf[i, j + 1], hf[i + 1, j + 1]
    z = np.where(u >= v, (1-u)*a + (u-v)*b + v*d, (1-v)*a + (v-u)*c + u*d)
    return np.where(valid, z, np.nan)


def finite_median(values):
    v = np.asarray(values)
    return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else None


def family(name):
    # Conservative grouping of known augmentation suffixes, not a provenance guarantee.
    return re.split(r'_aug\d*|_flipped', name)[0]


def analyze(m, terrain, min_support):
    fps = float(m['fps'])
    pos = fk(m)
    count = len(pos)
    out = dict(frames=count, fps=fps, duration_s=(count-1)/fps, events=[], score=0.,
               root_max_step_m=float(np.linalg.norm(np.diff(pos[:, 0], axis=0), axis=1).max()),
               terrain_present=terrain is not None)
    if 'body_contacts' not in m:
        return dict(out, category='missing_contacts')
    contacts = np.asarray(m['body_contacts'])
    if contacts.shape != (count, 15) or not np.isfinite(contacts).all():
        raise ValueError('Expected finite body_contacts[T,15]')
    c = contacts > 0
    both = c[:, 5] & c[:, 8]
    feet = c[:, 11] | c[:, 14]
    out.update(both_hand_s=float(both.sum()/fps), both_hand_air_s=float((both & ~feet).sum()/fps))
    for start, end in intervals(both):
        if (end-start)/fps < min_support:
            continue
        before = max(0, start-round(.7*fps))
        after = min(count, end+round(1.5*fps))
        anchor = pos[start:end, [5, 8], :2].mean((0, 1))
        direction = pos[min(count-1, end+round(.3*fps)), 0, :2] - pos[before, 0, :2]
        norm = np.linalg.norm(direction)
        direction = direction/max(norm, 1e-9)
        approach = float((pos[start, 0, :2]-anchor) @ direction)
        departure = float((pos[after-1, 0, :2]-anchor) @ direction)
        air = np.flatnonzero(~feet[start:end])
        air_frame = int(start+air[0]) if len(air) else None
        # Require an actual grounded-to-airborne transition within the support interval.
        simultaneous = bool(air_frame == start and start > 0 and feet[start-1])
        support_before_air = bool(air_frame is not None and air_frame > start
                                  and feet[start] and feet[air_frame-1])
        landing_candidates = np.flatnonzero(feet[(air_frame+1 if air_frame is not None else end):after])
        landing = int((air_frame+1 if air_frame is not None else end)+landing_candidates[0]) if len(landing_candidates) else None
        ground_before = heights(terrain, pos[before:start+1, [11,14], :2])
        base_height = finite_median(ground_before[c[before:start+1][:,[11,14]]])
        hand_z = heights(terrain, pos[start:end, [5,8], :2])
        hand_error = finite_median(abs(pos[start:end, [5,8], 2]-hand_z))
        surface_z = finite_median(hand_z)
        support_height = surface_z-base_height if surface_z is not None and base_height is not None else None
        landing_height = None
        landed_beyond = False
        if landing is not None:
            landing_z = finite_median(heights(terrain, pos[landing, [11,14], :2])[c[landing,[11,14]]])
            if landing_z is not None and base_height is not None:
                landing_height = landing_z-base_height
            landed_beyond = bool((pos[landing,0,:2]-anchor) @ direction > .15)
        elevated = support_height is not None and support_height > .15
        aligned = hand_error is not None and hand_error < .18
        crossed = approach < -.1 and departure > .15 and norm > .2
        if support_before_air and elevated and aligned and crossed and landed_beyond and landing_height is not None and abs(landing_height) < .3:
            category = 'two_hand_crossing_candidate'
        elif support_before_air and elevated and aligned and landing_height is not None and landing_height > .3:
            category = 'platform_ascent_candidate'
        elif len(air) and not support_before_air and not simultaneous:
            category = 'late_hand_support'
        else:
            category = 'two_hand_support_uncertain'
        score = (30*support_before_air + 15*elevated + 15*aligned + 15*crossed
                 + 10*landed_beyond + min((end-start)/fps, .8)*10 + min(len(air)/fps,.4)*10)
        out['events'].append(dict(start_frame=start, end_frame_exclusive=end, start_s=start/fps,
            end_s=end/fps, duration_s=(end-start)/fps, airborne_overlap_s=len(air)/fps,
            support_before_air=support_before_air, contact_and_takeoff_same_frame=simultaneous,
            landing_frame=landing,
            hand_surface_error_m=hand_error, support_height_above_approach_m=support_height,
            landing_height_change_m=landing_height, root_at_contact_along_travel_m=approach,
            root_after_contact_along_travel_m=departure, landed_beyond=landed_beyond,
            category=category, score=round(score,2)))
    priority = {'two_hand_crossing_candidate':4,'platform_ascent_candidate':3,
                'two_hand_support_uncertain':2,'late_hand_support':1}
    if out['events']:
        best=max(out['events'],key=lambda e:(priority[e['category']],e['score']))
        out.update(category=best['category'],score=best['score'],best_event=best)
    else:
        out['category']='no_sustained_two_hand_support'
    return out


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def render_gallery(out, selected):
    template=Path(__file__).with_name('parc_review.html').read_text(encoding='utf-8')
    gallery=[]
    for record in selected:
        try:
            m, terrain=load_clip(Path(record['source']))
            pos=fk(m)
            # Preserve all motion/contact frames. Modest heightfields are embedded without simplification.
            item=dict(record,positions=np.round(pos,5).tolist(),contacts=(np.asarray(m['body_contacts'])>0).astype(int).tolist(),
                      parents=PARENTS,names=NAMES,terrain=None)
            if terrain is not None:
                item['terrain']={'hf':np.round(terrain['hf'],5).tolist(),'dx':float(terrain['dx']),
                                 'min_point':np.asarray(terrain['min_point']).tolist()}
            gallery.append(item)
        except Exception as exc:
            print(f'Preview skipped: {record["clip"]}: {exc}',flush=True)
    payload=json.dumps({'clips':gallery,'labels':LABELS},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')
    return len(gallery)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('/root/gpufree-data/datasets/parc_ms'))
    parser.add_argument('--output',type=Path,default=Path('/root/gpufree-data/datasets/parc_review'))
    parser.add_argument('--min-support',type=float,default=.1,help='Minimum contiguous two-hand contact in seconds')
    parser.add_argument('--previews',type=int,default=24)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    if not args.input.is_dir():parser.error('Input directory does not exist')
    if args.min_support<=0 or args.previews<1:parser.error('Positive min-support and previews required')
    args.output.mkdir(parents=True,exist_ok=True)
    cache={}
    index=args.output/'index.jsonl'
    if index.exists() and not args.refresh:
        for line in index.read_text().splitlines():
            try:
                item=json.loads(line);cache[item['source']]=item
            except (ValueError,KeyError):pass
    files=sorted(args.input.rglob('*.pkl'))
    records=[]
    reused=0
    for i,path in enumerate(files):
        if '.cache' in path.parts:continue
        stat=path.stat()
        signature=[VERSION,stat.st_size,stat.st_mtime_ns,args.min_support]
        source=str(path.resolve())
        old=cache.get(source)
        if old and old.get('signature')==signature and old['category']!='error':
            record=old;reused+=1
        else:
            record=dict(source=source,clip=path.stem,family=family(path.stem),signature=signature)
            try:
                record.update(analyze(*load_clip(path),args.min_support))
                if path.stat().st_size!=stat.st_size or path.stat().st_mtime_ns!=stat.st_mtime_ns:
                    raise ValueError('Input changed during scan; retry on next run')
            except Exception as exc:
                record.update(category='error',score=0.,error=str(exc))
        record['terrain_obj']=str(path.with_name(path.stem+'_terrain.obj'))
        record['terrain_obj_present']=Path(record['terrain_obj']).is_file()
        records.append(record)
        if (i+1)%500==0:print(f'Scanned {i+1}/{len(files)}; reused {reused}',flush=True)
    tmp=index.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
    tmp.replace(index)
    candidates=sorted([r for r in records if r.get('events')],key=lambda r:r['score'],reverse=True)
    write_json(args.output/'candidates.json',candidates)
    fields=['clip','family','category','score','duration_s','both_hand_s','both_hand_air_s','terrain_obj_present','source']
    with (args.output/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(candidates)
    # Round-robin categories + at most two per family keep the first review page diverse.
    pools={cat:[r for r in candidates if r['category']==cat] for cat in
           ['two_hand_crossing_candidate','platform_ascent_candidate','two_hand_support_uncertain','late_hand_support']}
    selected=[];families=Counter()
    while len(selected)<args.previews and any(pools.values()):
        for pool in pools.values():
            while pool and families[pool[0]['family']]>=2:pool.pop(0)
            if pool and len(selected)<args.previews:
                r=pool.pop(0);selected.append(r);families[r['family']]+=1
    previews=render_gallery(args.output,selected)
    summary=dict(scanned_at_utc=datetime.now(timezone.utc).isoformat(),input=str(args.input.resolve()),
                 scanned_files=len(records),reused=reused,candidates=len(candidates),previews=previews,
                 categories=dict(Counter(r['category'] for r in records)),min_support_s=args.min_support,
                 note='Snapshot of completed local .pkl files; download may still be running. All classes are heuristic candidates, not confirmed vault labels. Contacts indicate contact, not force or object identity.')
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(f'Open {args.output / "index.html"}',flush=True)


if __name__=='__main__':main()
