"""Checks for temporal false positives, rig convention, and terrain interpolation."""
import io
import pickle
from unittest.mock import patch

import numpy as np
import pytest

import parc_catalog as pc


def test_intervals_are_contiguous_and_end_exclusive():
    assert pc.intervals([1, 1, 0, 1, 0]) == [(0, 2), (3, 4)]
    assert pc.intervals([]) == []


def test_fk_rotates_children_and_preserves_bone_lengths():
    m = {'root_pos': np.array([[1., 2., 3.]]), 'root_rot': np.array([[0, 0, 2**-.5, 2**-.5]]),
         'joint_rot': np.tile([0., 0., 0., 1.], (1, 14, 1))}
    pos = pc.fk(m)
    assert np.allclose(pos[0, 9], [1+.084887, 2, 3])
    for j in range(1, 15):
        assert np.isclose(np.linalg.norm(pos[0, j]-pos[0, pc.PARENTS[j]]), np.linalg.norm(pc.OFFSETS[j]))


def test_triangle_heightfield_orientation_and_out_of_bounds():
    terrain = {'hf': np.array([[0., 1.], [2., 4.]]), 'dx': 1., 'min_point': np.array([10., 20.])}
    result = pc.heights(terrain, [[10.75, 20.25], [10.25, 20.75], [11., 21.], [12., 20.]])
    assert np.allclose(result[:3], [2., 1.5, 4.])
    assert np.isnan(result[3])


def fixture(late=False, platform=False):
    # Hands support a one-metre obstacle before the root crosses its centre.
    t=30
    p=np.zeros((t,15,3)); p[:,0,0]=np.linspace(-1,1.5,t);p[:,0,2]=1.
    p[:,[5,8],2]=1.
    p[:,[11,14],0]=p[:,0,0,None]
    c=np.zeros((t,15));c[:,[11,14]]=1
    c[8:16,[5,8]]=1
    c[5 if late else 10:20,[11,14]]=0
    m={'fps':10.,'root_pos':p[:,0], 'body_contacts':c}
    terrain={'hf':np.zeros((8,8)), 'dx':1., 'min_point':np.array([-3.,-3.])}
    def height(_,xy):
        arr=np.asarray(xy);x=arr[...,0]
        return np.where(abs(x)<.15,1.,np.where((x>.2)&platform,1.,0.))
    return m,terrain,p,height


def test_support_then_air_crossing():
    m,t,p,h=fixture()
    with patch.object(pc,'fk',return_value=p),patch.object(pc,'heights',side_effect=h):
        r=pc.analyze(m,t,.1)
    assert r['category']=='two_hand_crossing_candidate'
    assert r['best_event']['support_before_air']


def test_air_before_hands_is_not_target():
    m,t,p,h=fixture(late=True)
    with patch.object(pc,'fk',return_value=p),patch.object(pc,'heights',side_effect=h):
        r=pc.analyze(m,t,.1)
    assert r['category']=='late_hand_support'


def test_same_frame_hand_contact_and_takeoff_is_ambiguous():
    m,t,p,h=fixture()
    m['body_contacts'][8:10,[11,14]]=0
    with patch.object(pc,'fk',return_value=p),patch.object(pc,'heights',side_effect=h):
        r=pc.analyze(m,t,.1)
    assert r['category']=='two_hand_support_uncertain'
    assert r['best_event']['contact_and_takeoff_same_frame']
    assert not r['best_event']['support_before_air']


def test_landing_on_top_is_platform_ascent():
    m,t,p,h=fixture(platform=True)
    with patch.object(pc,'fk',return_value=p),patch.object(pc,'heights',side_effect=h):
        r=pc.analyze(m,t,.1)
    assert r['category']=='platform_ascent_candidate'


def test_restricted_pickle_rejects_arbitrary_globals():
    with pytest.raises(ValueError):
        pc.decode(pickle.dumps(eval))
    assert np.array_equal(pc.decode(pickle.dumps(np.arange(4))), np.arange(4))


def test_absent_contacts_not_silently_zero_contacts():
    m,t,p,h=fixture();del m['body_contacts']
    with patch.object(pc,'fk',return_value=p):
        assert pc.analyze(m,t,.1)['category']=='missing_contacts'
