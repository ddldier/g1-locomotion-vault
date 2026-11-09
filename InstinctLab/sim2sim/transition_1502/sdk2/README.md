# Unitree SDK2 C++ 路径

这个目录是 `model_13507.pt` 的 Unitree SDK2 接口探针。它使用官方 SDK2
订阅 G1 的 `rt/lowstate`，使用 ONNX Runtime 执行 actor，并通过本机 UDP
接收 527 点高度扫描。当前程序是**只读 dry run**：没有创建
`rt/lowcmd` publisher，也不会给机器人发送任何电机命令。

## 已验证的文件

```text
模型：/root/gpufree-data/InstinctMJ/sim2sim/transition_1502/artifacts/model_13507_actor.onnx
程序：/root/gpufree-data/InstinctMJ/sim2sim/transition_1502/sdk2/build/g1_transition_sdk2_probe
SDK2：/root/gpufree-data/unitree_sdk2/install
官方仿真桥：/root/gpufree-data/unitree_mujoco/simulate/build/unitree_mujoco
```

## 编译和自检

```bash
cd /root/gpufree-data/InstinctMJ
cmake -S sim2sim/transition_1502/sdk2 \
  -B sim2sim/transition_1502/sdk2/build
cmake --build sim2sim/transition_1502/sdk2/build -j2

sim2sim/transition_1502/sdk2/build/g1_transition_sdk2_probe \
  sim2sim/transition_1502/artifacts/model_13507_actor.onnx --self-test
```

自检会确认 ONNX 是 `[batch,2049] -> [batch,29]`，Unitree 的 29 个 motor
ID 没有重复，并执行一次 C++ 推理。

## 与官方 MuJoCo SDK2 桥联调

终端 1（官方桥）:

```bash
cd /root/gpufree-data/unitree_mujoco
simulate/build/unitree_mujoco -r g1 -s scene_29dof.xml -n lo -i 0
```

终端 2（探针）:

```bash
cd /root/gpufree-data/InstinctMJ
sim2sim/transition_1502/sdk2/build/g1_transition_sdk2_probe \
  sim2sim/transition_1502/artifacts/model_13507_actor.onnx \
  --network lo 5556
```

探针只会在同时收到新鲜的 SDK2 `LowState` 和 UDP 扫描时推理。`LowState`
超时、扫描超过 100 ms 未更新、或 DDS `tick` 停止变化时会暂停；恢复后会
重新填充历史帧，避免把中断前的旧姿态混入新输入。

## 高度扫描 UDP 契约

每个数据报必须是恰好 527 个 little-endian `float32`，即 2108 字节，发送到
`127.0.0.1:5556`（可用最后一个命令行参数改端口）。527 点对应训练中的
`31 x 17` 扫描顺序；数值应当是训练传感器使用的相对高度，发送前按
`[-1, 1]` 截断。程序会拒绝长度不等于 2108 字节的数据报，以及包含 NaN 或
无穷大的扫描。

当前官方 `unitree_mujoco` G1 场景没有 InstinctLab rev10 的箱体和训练时
深度传感器，因此此前联调使用常数扫描只验证 DDS、UDP 和 ONNX 接口，不能
证明 vault 策略已经完成 sim2sim 迁移。要验证动作，需要加入与训练一致的
raycast/深度到 527 点扫描转换，并使用相同的 pelvis-root rev10 机器人模型。

## 观测和关节映射

输入顺序固定为：

```text
高度历史 527 x 3
速度命令 3
投影重力 3 x 5
pelvis 角速度 3 x 5
关节位置（q - default）29 x 5
关节速度 29 x 5
上次动作 29 x 5
```

历史按旧到新排列；首次输入会复制当前样本填满历史。动作输出仍是 29 个
关节位置目标的策略空间值。代码中的 `kMotorIds` 按策略关节顺序映射到
Unitree SDK2 官方 G1 29DoF 电机 ID，不能改成简单的数组下标对应。

## 接真机前仍必须完成的工作

1. 用真实 G1 的 SDK2 `LowState` 核对 IMU 安装轴、四元数顺序、角速度单位和
   pelvis 坐标系；仿真桥的坐标不能直接当作真机标定结果。
2. 把真实深度/激光传感器转换为上面的 527 点局部高度图，并确认时间戳、
   yaw 对齐、缺测处理和裁剪范围。
3. 逐项核对 29 个关节的顺序、零位、方向、减速比、软限位和默认姿态。
4. 根据实机辨识设置 PD 增益、动作缩放、力矩限制、通信周期和 watchdog。
5. 增加 FSM、低电量/通信丢失保护、姿态跌倒保护和物理急停；先在吊架和低
   增益条件下测试。
6. 经过以上检查后，才可以另行实现受限的 `rt/lowcmd` publisher。当前探针
   刻意不包含该部分，不能直接用于让真机运动。

## 与旧 Python MuJoCo 版本的关系

`../mujoco_udp_sim.py` 是独立的 Python MuJoCo 播放器，走 UDP policy server；
本目录的 C++ 程序走官方 SDK2 DDS `LowState`。两者共享 ONNX 观测约定，但
不是同一个动力学环境。修改训练模型或观测配置后，必须重新导出 ONNX，并
重新运行 `--self-test` 与仿真桥联调。
