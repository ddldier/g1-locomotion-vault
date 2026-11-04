"""Pure Torch transition bookkeeping; no Isaac/Omniverse imports."""
# 模块级文档字符串：说明本模块是"纯 Torch 的过渡任务簿记（bookkeeping）"，
# 不导入任何 Isaac / Omniverse 相关模块，因此可以在无仿真器的环境下单独测试与复用。
# 主要职责：管理每个并行环境在"过渡（transition）任务"中的状态机（是否遇到障碍、是否 RSI、
# 是否越过箱子、是否完成/失败等），以及各类诊断缓冲区。

from dataclasses import dataclass   # 用 dataclass 简洁地定义配置容器
import torch                        # 唯一的计算依赖


# ---------------------------------------------------------------------------
# 1) 过渡任务的配置（超参数集合）
# 用 @dataclass 定义，所有字段都有默认值，便于用 dataclasses.replace 做局部覆盖。
# 注意：distances 与 frame_range 用 tuple 而非 list，保证 hashable / 不可变语义。
# ---------------------------------------------------------------------------
@dataclass
class TransitionSettings:
    obstacle_probability: float = 0.8        # 每个 episode 出现障碍物的概率
    rsi_probability: float = 0.6             # 触发 RSI（随机状态初始化）的概率
    distances: tuple = (0.8, 1.5, 2.2)       # 箱子放置距离的候选集合（单位：米）
    recovery_hold_s: float = 0.6             # 稳定恢复需要持续的时间（秒），才判定"完成"
    lane_half_width: float = 1.0             # 赛道半宽（y 方向 ±lane_half_width 视为在赛道内）
    approach_distance: float = 2.5           # "接近段"判定距离（箱子前方多远开始算接近）
    vault_entry_distance: float = 0.7        # "翻越段"判定距离（箱子前方多近开始算进入 vault）
    rear_margin: float = 0.15                # 越过箱子后需要额外超出后沿的余量
    min_base_height: float = 0.35            # 基座最低允许高度，低于此值判定失败
    max_episode_s: float = 12.0              # 单个 episode 最长时长（秒）
    forward_command: float = 1.0             # 默认前进速度指令（m/s）
    box_force_threshold: float = 40.0        # 判定"非法箱体接触"的力阈值（N）
    ground_force_threshold: float = 40.0     # 判定"非法地面接触"的力阈值（N）
    motion_file: str = ""                    # 参考运动文件路径（RSI 时使用，可为空）
    frame_range: tuple = (102, 665)          # 参考运动的帧区间（起止帧）

    def validate(self):
        """配置合法性检查；在 TransitionState 构造时调用。"""
        # 两个概率必须落在 [0,1]
        if not 0 <= self.obstacle_probability <= 1 or not 0 <= self.rsi_probability <= 1:
            raise ValueError("Sampling probabilities must be in [0,1].")
        # 距离集合非空且全为正
        if not self.distances or min(self.distances) <= 0:
            raise ValueError("Distances must be nonempty and positive.")
        # 恢复时长必须为正，且 episode 总时长要严格大于恢复时长（否则永远无法完成）
        if self.recovery_hold_s <= 0 or self.max_episode_s <= self.recovery_hold_s:
            raise ValueError("Invalid recovery/episode durations.")


# ---------------------------------------------------------------------------
# 2) 四元数乘法（Hamilton 积）
# 输入 a, b 形状均为 (..., 4)，排列为 (w, x, y, z)。
# 返回 a ⊗ b，形状同为 (..., 4)。用于复合两个旋转。
# ---------------------------------------------------------------------------
def quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)   # 拆分 a 的分量
    bw, bx, by, bz = b.unbind(-1)   # 拆分 b 的分量
    # 标准 Hamilton 四元数乘法公式
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,   # w 分量
                        aw*bx+ax*bw+ay*bz-az*by,   # x 分量
                        aw*by-ax*bz+ay*bw+az*bx,   # y 分量
                        aw*bz+ax*by-ay*bx+az*bw), -1)  # z 分量


# ---------------------------------------------------------------------------
# 3) 将世界坐标点转换到"仅考虑 yaw 的箱子局部坐标系"
# 输入：
#   points: (N,3) 或 (N,B,3)，每行是一个世界坐标点
#   center: (N,3)，箱子中心的世界坐标
#   quat:   (N,4)，箱子朝向（取其中的 yaw 分量）
# 输出：
#   与 points 同形状的局部坐标 (x', y', z')，其中 z' 就是世界 z 减去中心 z
# 说明：忽略 roll/pitch，只做绕 z 轴的旋转平移，因此适合"贴地箱子"的判定。
# ---------------------------------------------------------------------------
def box_local(points, center, quat):
    """Yaw-only box coordinates, supporting (N,3) and (N,B,3)."""
    # 从四元数提取 yaw 角：yaw = atan2(2(wz+xy), 1-2(y²+z²))
    yaw = torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),
                      1-2*(quat[:,2].square()+quat[:,3].square()))
    # 平移：若 points 是三维（含 B 个点），center 需要广播到第二维
    d = points - (center[:,None,:] if points.ndim == 3 else center)
    c, s = yaw.cos(), yaw.sin()
    if points.ndim == 3:
        # 广播到 (N,1)，与 (N,B,3) 对齐
        c, s = c[:,None], s[:,None]
    # 绕 z 轴旋转 -yaw（即把世界方向转到箱子局部方向）
    return torch.stack((c*d[...,0]+s*d[...,1],   # x'
                        -s*d[...,0]+c*d[...,1],  # y'
                        d[...,2]), -1)           # z'（未旋转）


# ---------------------------------------------------------------------------
# 4) 过渡任务的核心状态机
# 为每个并行环境维护一组 bool / float / long 张量，逐 step 更新。
# 关键约定：
#   - 所有"逐环境"张量形状都是 (num_envs,)，可以直接用 ids 做掩码索引。
#   - reset 会把指定 ids 的所有同形状张量清零，然后按 has_obstacle / is_rsi 重新初始化。
#   - update 通过 step 号去重（stamp），保证同一 step 对同一环境只更新一次。
# ---------------------------------------------------------------------------
class TransitionState:
    def __init__(self, num_envs, device, settings):
        settings.validate()                    # 先校验配置
        self.cfg = settings                    # 保存配置
        self.device = device                   # 计算设备（cpu / cuda）
        self.n = num_envs                      # 并行环境数量

        # 4.1) 布尔状态位（每个环境一个 bool）
        for name in ("has_obstacle",        # 本 episode 是否有障碍物
                     "is_rsi",              # 本 episode 是否触发 RSI
                     "full_start",          # 是否为"完整起点"（有障碍且非 RSI，即从标准起点开始）
                     "recovery_seed",       # 是否为"恢复起点"（有障碍且 RSI，且初始 x 在箱子后方）
                     "ever_front",          # 是否曾经位于箱子前方（且当时在赛道内）
                     "seen_over",           # 是否曾从箱子上方"看到/越过"（高度超过箱顶+0.18）
                     "touched_box",         # 是否发生过合法的箱体接触（踩上箱子）
                     "crossed",             # 是否成功越过箱子后沿
                     "completed",           # 是否完成恢复稳定（满足恢复保持时长）
                     "failed",              # 是否已判定失败
                     "lane_violation",      # 是否发生赛道外偏离（接近/通过阶段）
                     "no_obstacle_excursion"):  # 无障碍 episode 是否发生异常（高抬升/倾斜）
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool, device=device))

        # 4.2) 浮点状态位（每个环境一个 float）
        for name in ("prev_x",              # 上一步的 relative x（用于计算速度/交叉）
                     "best_x",              # 历史最大 x（用于计算前进增益）
                     "recovery_timer",      # 稳定恢复计时器（秒）
                     "stall_timer",         # 停滞计时器（秒）
                     "approach_rate",       # 接近段的瞬时前进速率（m/s）
                     "progress_rate",       # 翻越段的瞬时前进速率（m/s）
                     "speed_reward"):        # 速度奖励（仅非翻越段计算）
            setattr(self, name, torch.zeros(num_envs, device=device))

        # 4.3) 离散状态：stage 表示当前所处阶段
        #     0 = 尚未接近；1 = 接近段；2 = 翻越段；3 = 已越过/恢复种子
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 4.4) stamp：记录每个环境最近一次 update 时的 step 号，
        #      用于"同一 step 只更新一次"（避免 reset 后的第一次观测也推进计时器）
        self.stamp = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # 4.5) 诊断缓冲区：只用于报告/分析，绝不反馈到策略或任务决策
        self.failure_code = torch.zeros(num_envs, dtype=torch.long, device=device)  # 失败原因位掩码
        self.terminal_box_body_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)     # 非法箱体接触的刚体索引
        self.terminal_ground_body_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)  # 非法地面接触的刚体索引
        for name in ("start_front_distance",           # 初始时相对箱子前沿的前方距离
                     "terminal_base_height",           # 末态基座高度
                     "terminal_gravity_z",             # 末态重力方向 z 分量（用于判断直立）
                     "terminal_relative_x",            # 末态相对 x
                     "terminal_relative_y",            # 末态相对 y
                     "terminal_illegal_box_force",     # 末态非法箱体接触力
                     "terminal_illegal_ground_force"): # 末态非法地面接触力
            setattr(self, name, torch.zeros(num_envs, device=device))

        # 4.6) 事件标志（每个 step 会重新计算；作为"当前 step 是否发生"的瞬时信号）
        self.cross_event = torch.zeros(num_envs, dtype=torch.bool, device=device)     # 本 step 发生越过后沿
        self.complete_event = self.cross_event.clone()   # 本 step 发生完成
        self.full_success = self.cross_event.clone()     # 本 step 发生"完整起点 + 越过"的成功
        self.failure_event = self.cross_event.clone()    # 本 step 首次判定失败

    def reset(self, ids, has_obstacle, is_rsi, initial_x, half_length, step):
        """
        重置指定环境的状态。
        参数：
          ids:           要重置的环境索引
          has_obstacle:  本次是否放置障碍物
          is_rsi:        本次是否 RSI
          initial_x:     初始相对 x（相对于箱子中心）
          half_length:   箱子半长（沿 x 方向）
          step:          当前仿真 step 号（写入 stamp，避免同一 step 内重复推进）
        """
        # 5.1) 把所有形状为 (n,) 的张量在 ids 上清零（布尔、浮点、long 都适用）
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.shape == (self.n,):
                value[ids] = 0

        # 5.2) 根据采样结果设置基本标志
        self.has_obstacle[ids] = has_obstacle
        self.is_rsi[ids] = is_rsi
        # "完整起点"= 有障碍且非 RSI（即从标准起点直接跑）
        self.full_start[ids] = has_obstacle & ~is_rsi
        # "曾经在箱子前方"= 有障碍且初始 x 小于前沿（说明从箱子前方出发）
        self.ever_front[ids] = has_obstacle & (initial_x < -half_length)
        # "恢复种子"= 有障碍 + RSI + 初始 x 已经在箱子后方（从越过后开始，向稳定恢复过渡）
        self.recovery_seed[ids] = has_obstacle & is_rsi & (initial_x > half_length)

        # 5.3) 初始化用于速度/增益计算的量
        self.prev_x[ids] = initial_x
        self.best_x[ids] = initial_x
        # 初始时相对前沿的前方距离（正数表示在箱子前方多远）
        self.start_front_distance[ids] = -half_length-initial_x

        # 5.4) 诊断缓冲区恢复为"无效"值
        self.terminal_box_body_index[ids] = -1
        self.terminal_ground_body_index[ids] = -1

        # 5.5) 写 stamp：本 step 内 update 会跳过这些环境，避免 reset 当步就推进计时器
        self.stamp[ids] = step

        # 5.6) 阶段初始化：恢复种子直接进入阶段 3，否则从 0 开始
        self.stage[ids] = torch.where(self.recovery_seed[ids], 3, 0)

    def update(self, step, dt, *, relative_base, half_length, box_top,
               base_height, gravity_z, lin_vel, ang_vel, command,
               feet_ground, all_links_clear, legal_box_contact,
               illegal_box_contact, illegal_ground_contact):
        """
        每个仿真 step 调用一次，更新所有环境的过渡状态。
        参数（除 step/dt 外均为逐环境张量，形状 (N,) 或 (N,K)）：
          relative_base:          (N,3) 基座相对箱子的位置
          half_length:            (N,)  箱子半长（x 方向）
          box_top:                (N,)  箱顶高度
          base_height:            (N,)  基座高度
          gravity_z:              (N,)  重力方向 z 分量（-1 表示完全直立）
          lin_vel:                (N,3) 基座线速度
          ang_vel:                (N,3) 基座角速度
          command:                (N,3) 速度指令（x 前进，z 转向）
          feet_ground:            (N,)  双脚是否着地
          all_links_clear:        (N,)  所有连杆是否"干净"（无非法接触）
          legal_box_contact:      (N,)  是否发生合法箱体接触（允许踩箱子）
          illegal_box_contact:    (N,)  是否发生非法箱体接触（撞击/侧撞）
          illegal_ground_contact: (N,)  是否发生非法地面接触（除脚以外的部位触地）
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")   # dt 必须为正

        # 6.1) 只对"本 step 尚未更新"的环境执行（stamp 去重）
        active = self.stamp != step
        if not active.any():
            return

        # 6.2) 拆解常用量
        h = self.has_obstacle                       # 是否有障碍
        x, y = relative_base[:,0], relative_base[:,1]   # 相对箱子的 x / y
        front = -half_length                        # 箱子前沿（相对坐标）
        rear = half_length                          # 箱子后沿（相对坐标）
        in_lane = y.abs() <= self.cfg.lane_half_width   # 是否在赛道内
        overlaps = (x >= front) & (x <= rear)       # x 方向与箱子重叠
        near = (x >= front-self.cfg.approach_distance) & (x <= rear+self.cfg.rear_margin)  # 处于"接近/通过"范围

        # 6.3) 赛道偏离：在接近/通过阶段跑出赛道，则本 attempt 永久失效
        self.lane_violation |= active & h & near & ~in_lane

        # 6.4) 记录"曾经在前方"（在赛道内、x 小于前沿）
        self.ever_front |= active & h & (x < front) & in_lane

        # 6.5) 记录"曾从箱子上方越过"（高度超过箱顶 + 0.18）
        self.seen_over |= active & h & overlaps & in_lane & (base_height > box_top+0.18)

        # 6.6) 记录"合法踩箱"（重叠区域内、在赛道内、且发生合法箱体接触）
        self.touched_box |= active & h & overlaps & in_lane & legal_box_contact

        # 6.7) 失败条件（任一满足即失败）
        new_failure = ((base_height < self.cfg.min_base_height)   # 基座太低
                       | (gravity_z > -0.15)                     # 倾斜过大（重力 z 不够负）
                       | illegal_ground_contact                  # 非法地面接触
                       | (h & illegal_box_contact)               # 非法箱体接触（仅在有障碍时）
                       | self.lane_violation)                    # 赛道偏离

        # 6.8) 位掩码编码失败原因（同一 episode 可能多个原因同时成立，因此用位或）
        #      位定义：1 = 低基座；2 = 倾斜；4 = 连杆触地；8 = 撞箱；16 = 偏离赛道
        first_failure = active & new_failure & ~self.failed   # 首次失败（用于记录 failure_code）
        code = ((base_height < self.cfg.min_base_height).long()
                + 2*(gravity_z > -0.15).long()
                + 4*illegal_ground_contact.long()
                + 8*(h & illegal_box_contact).long()
                + 16*self.lane_violation.long())
        self.failure_code[first_failure] = code[first_failure]   # 只在首次失败时写入
        self.failure_event[active] = new_failure[active] & ~self.failed[active]   # 本 step 的失败事件
        self.failed |= active & new_failure                       # 累积失败标记

        # 6.9) 判定"越过箱子后沿"
        #      条件：有障碍、曾在前方、尚未越过、未失败、未偏离赛道、
        #            上一步还在后沿余量内、本步已超出后沿余量、在赛道内、
        #            且曾经从上方越过或踩过箱子（确认真实翻越而非绕行）
        crosses = (h & self.ever_front & ~self.crossed & ~self.failed & ~self.lane_violation
                   & (self.prev_x <= rear+self.cfg.rear_margin)
                   & (x > rear+self.cfg.rear_margin)
                   & in_lane & (self.seen_over | self.touched_box))
        self.cross_event[active] = crosses[active]
        self.crossed |= active & crosses

        # 6.10) 判定"稳定恢复"（可用于判定完成）
        #       eligible：已越过箱子，或本身是恢复种子
        #       stable  ：未失败、所有连杆干净、双脚着地、足够直立、
        #                 线速度与指令误差小、角速度 z 与指令误差小、总角速度小
        eligible = self.crossed | self.recovery_seed
        speed_error = torch.linalg.vector_norm(lin_vel[:,:2]-command[:,:2], dim=-1)
        stable = (eligible & ~self.failed & all_links_clear & feet_ground
                  & (gravity_z < -0.8)                          # 足够直立
                  & (speed_error < 0.6)                         # 线速度接近指令
                  & ((ang_vel[:,2]-command[:,2]).abs() < 0.8)   # 转向角速度接近指令
                  & (torch.linalg.vector_norm(ang_vel,dim=-1) < 2.0))  # 总角速度不过大

        # 6.11) 恢复计时器：稳定则累加，不稳定则清零
        timer = torch.where(stable, self.recovery_timer+dt, torch.zeros_like(x))
        self.recovery_timer[active] = timer[active]

        # 6.12) 完成判定：eligible + 未完成 + 未失败 + 计时器达到保持时长
        completion = eligible & ~self.completed & ~self.failed & (timer >= self.cfg.recovery_hold_s)
        self.complete_event[active] = completion[active]
        self.completed |= active & completion
        # "完整成功"= 完整起点 + 已越过 + 完成
        self.full_success |= active & completion & self.full_start & self.crossed

        # 6.13) 阶段更新（0/1/2/3）
        stage = torch.zeros_like(self.stage)
        stage[h & (x >= front-self.cfg.approach_distance)] = 1   # 进入接近段
        stage[h & (x >= front-self.cfg.vault_entry_distance)] = 2  # 进入翻越段
        stage[h & eligible] = 3                                   # 已越过或恢复种子
        self.stage[active] = stage[active]

        # 6.14) 接近段前进速率（仅当在接近段、尚未失败、在赛道内时记录）
        approach = h & (self.prev_x < front) & (x < front) & in_lane & ~self.failed
        delta = (x-self.prev_x).clamp(-0.2,0.2)   # 限制单步位移，防止异常跳变
        self.approach_rate[active] = torch.where(approach,delta/dt,0.)[active]

        # 6.15) 翻越段前进增益（capped 到 rear+0.8，避免过度奖励）
        capped = torch.minimum(x,rear+0.8)
        gain = (capped-self.best_x).clamp(0.,0.2)   # 单步最大增益 0.2
        progress = h & (stage==2) & ~self.failed & in_lane
        self.progress_rate[active] = torch.where(progress,gain/dt,0.)[active]
        self.best_x[active] = torch.maximum(self.best_x,capped)[active]

        # 6.16) 停滞计时器：本步有明显增益则清零，否则累加
        stall = torch.where(gain>0.001,0.,self.stall_timer+dt)
        self.stall_timer[active] = stall[active]

        # 6.17) 速度奖励（仅在非翻越段 stage!=2 时计算）
        #      速度奖励：一个较宽的速度带，越接近 (speed_error=0.35, 转向误差=0) 越好
        speed = torch.exp(-((speed_error-0.35).clamp_min(0.)/0.7).square()
                          -((ang_vel[:,2]-command[:,2]).abs()/0.8).square())
        self.speed_reward[active] = torch.where(stage!=2,speed,0.)[active]

        # 6.18) 无障碍 episode 的异常判定：基座过高或倾斜过大
        self.no_obstacle_excursion |= active & ~h & ((base_height>0.95)|(gravity_z>-0.65))

        # 6.19) 更新 prev_x 与末态诊断量
        self.prev_x[active] = x[active]
        self.terminal_base_height[active] = base_height[active]
        self.terminal_gravity_z[active] = gravity_z[active]
        self.terminal_relative_x[active] = x[active]
        self.terminal_relative_y[active] = y[active]

        # 6.20) 写 stamp，标记本 step 已处理
        self.stamp[active] = step

    def records(self, ids):
        """
        导出指定环境的记录（用于 episode 结束时写入报告）。
        返回 dict：每个字段都是 (len(ids), ...) 的 detach 副本，
        这样即使后续环境继续更新也不会影响已导出的记录。
        """
        return {name: getattr(self,name)[ids].detach().clone() for name in
                ("has_obstacle", "is_rsi", "full_start", "crossed", "completed", "full_success",
                 "failed", "no_obstacle_excursion", "failure_code", "stage",
                 "start_front_distance", "terminal_base_height", "terminal_gravity_z",
                 "terminal_relative_x", "terminal_relative_y", "terminal_illegal_box_force",
                 "terminal_illegal_ground_force", "terminal_box_body_index", "terminal_ground_body_index")}
