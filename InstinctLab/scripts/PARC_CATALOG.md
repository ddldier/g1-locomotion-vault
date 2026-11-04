# PARC-MS 动作索引与双手支撑审阅

在现有 isaaclab Python 环境中运行，不启动 Isaac Sim。依赖 NumPy、SciPy。

```bash
cd /root/gpufree-data/InstinctLab
python scripts/parc_catalog.py
```

默认读取 `/root/gpufree-data/datasets/parc_ms`，结果写入数据盘
`/root/gpufree-data/datasets/parc_review`。脚本不修改原始数据，也不下载数据。
下载未完成时可以先扫描；再次执行会复用未变文件的结果，处理新增文件。
扫描使用启动时的本地文件快照，下载完成后应再运行一次。

## 输出

- `index.jsonl`：所有已扫描动作，包含来源、时长、接触区间、分类和错误。
- `candidates.json` / `candidates.csv`：有持续双手接触的全部候选。
- `summary.json`：本轮时间、数量、分类、缓存复用情况。
- `index.html`：完全离线的交互页面，不依赖 CDN 或 HTTP 服务。

双击 `index.html`，可以搜索、按类型过滤、慢放、逐帧、旋转和缩放。
绿色圈表示接触标记；时间线依次为右手、左手、右脚、左脚。
页面默认选取 24 个代表样本，各类轮流选取，每个推测增强家族最多两条。
原始坐标、帧率和所有帧保持不变，地形来自 pkl 内的完整高度场。

人工判断会尝试保存在当前浏览器的 localStorage，建议点击“导出本页人工审阅记录”
下载 JSON 保存。记录不会自动修改训练数据或成为已验证的训练标签。

```bash
# 展示更多代表候选
python scripts/parc_catalog.py --previews 60

# 要求至少 0.2 秒连续双手接触
python scripts/parc_catalog.py --min-support 0.2

# 忽略缓存，重新扫描
python scripts/parc_catalog.py --refresh
```

## 初筛规则及限制

骨架采用 PARC 的 15-body DFS 顺序，手为索引 5/8，脚为 11/14；四元数 xyzw，
Z 向上，单位米。运动学约定及骨架参数对应
`https://github.com/Roboparty/human-humanoid-tools/blob/main/hhtools/io/parc_ms_skeleton.py`
（Apache-2.0）和上游 PARC `humanoid.xml`。

默认持续双手接触阈值 0.1 秒。对每个连续区间，记录脚离地、后续落地、
手部地形高度差、骨盆沿运动方向相对于支撑位置的变化。

- 双手支撑越障候选：支撑开始时至少一只脚接触；之后区间内双脚离地；
  支撑面高于接近地面 0.15 米；手部节点高度误差中位数低于 0.18 米；
  骨盆由支撑点前方超过 0.1 米移动至后方超过 0.15 米；落地时也已越过；
  落地面与接近地面高度差小于 0.3 米。
- 撑上平台候选：先支撑后离地，支撑面和手部几何匹配，但落地面高出 0.3 米。
- 先腾空／后撑手：双手支撑期间有离地，但未看到先支撑再离地的时序。
- 双手支撑时序待核对：其余情况，包括同一帧才出现手接触和脚离地。

同帧接触不能分辨毫秒级先后，不能强行标为目标动作。接触数据不说明接触对象
或受力，手腕/脚踝节点也不等于真实碰撞面，粗糙高度场同样可能导致误判。
“跨过支撑点投影”不是“完整身体通过规则箱体”的充分条件，仍需人工查看。
规则采用短时间窗，慢动作、转弯、复杂多次支撑也可能漏检。

家族字段仅按 `_aug` / `_flipped` 命名推测，不能替代真实来源信息；
后续训练测试划分应进一步核对原始 motion 来源。

## 验证

```bash
python -m pytest scripts/test_parc_catalog.py -q
```

测试覆盖连续区间、骨架变换、三角网格高度插值、先支撑后离地、后撑手、
同帧歧义、平台落地、缺失接触标记及受限 pickle 读取。

后续接入 G1 前，需要人工确定样本、完成重定向、检查手部接触与碰撞，
再导出 `*_retargeted.npz` 和动作匹配地形的 `metadata.yaml`。
