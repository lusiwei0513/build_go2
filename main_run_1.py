import mujoco
import mujoco.viewer
import onnxruntime as ort
import numpy as np
import time

# 1. 配置区
MODEL_PATH = "scene2.xml"
POLICY_PATH = "policy.onnx"
ACTION_SCALE = 0.25 # 增加到0.25

# 1. 严格按照 Isaac 顺序排列的基准值 (FL, FR, RL, RR)
DEFAULT_DOF_POS_ISAAC = np.array([
    0.1, -0.1,  0.1, -0.1,   # Hips
    0.5,  0.5,  1.0,  1.0,   # Thighs (前腿0.8, 后腿1.0)
   -1.1, -1.1, -1.2, -1.2    # Calves
], dtype=np.float32)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
session = ort.InferenceSession(POLICY_PATH, providers=['CPUExecutionProvider'])

# 映射索引
mj_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_mj = np.argsort(mj_to_isaac)

OBS_DIM = 225
history_buffer = np.zeros(OBS_DIM, dtype=np.float32)
last_action_isaac = np.zeros(12, dtype=np.float32)

def get_single_frame_obs(data, commands, last_action_isaac):
    # A. 角速度 (3)
    ang_vel = data.sensor('imu_gyro').data.copy()
    
    # B. 重力投影 (3)
    q = data.sensor('imu_quat').data  # [w, x, y, z]
    gravity_proj = np.zeros(3)
    mujoco.mju_rotVecQuat(gravity_proj, np.array([0, 0, 1.0]), q)
    print(f"重力投影: {gravity_proj}")
    # C. 关节位置与速度 (重排为 Isaac 顺序)
    mj_qpos = data.qpos[7:19]
    mj_qvel = data.qvel[6:18]
    
    dof_pos_isaac = mj_qpos[mj_to_isaac] - DEFAULT_DOF_POS_ISAAC
    dof_vel_isaac = mj_qvel[mj_to_isaac]
    
    # D. 拼接当前帧 (45维)
    current_frame = np.concatenate([
        ang_vel,          # 3
        gravity_proj,     # 3
        commands,         # 3
        dof_pos_isaac,    # 12
        dof_vel_isaac,    # 12
        last_action_isaac # 12
    ])
    return current_frame

# 3. 仿真主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    # --- 初始预设站立环节 (1.5秒钟) ---
    print("正在初始化站立姿态，请等待1.5秒...")
    mj_home_ctrl = DEFAULT_DOF_POS_ISAAC[isaac_to_mj]
    
    # 设置初始关节位置
    data.qpos[7:19] = mj_home_ctrl
    mujoco.mj_forward(model, data)  # 更新一次状态
    
    # 平稳过渡到站立姿态
    for _ in range(750):
        data.ctrl[:] = mj_home_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    current_obs = get_single_frame_obs(data, np.array([0,0,0]), np.zeros(12))
    print(f"当前关节位置偏差: {current_obs[9:21]}")
    print("开始行走测试！")
    
    # --- 行走逻辑环节 ---
    control_dt = 0.02  # 50Hz控制频率（匹配Isaac Gym训练）
    steps_per_control = int(control_dt / model.opt.timestep) # 自动计算步数
    
    print(f"控制频率: {1/control_dt:.1f}Hz")
    print(f"物理步长: {model.opt.timestep*1000:.1f}ms")
    print(f"每控制周期步数: {steps_per_control}")

    step_count = 0
    while viewer.is_running():
        loop_start_time = time.time()
        
        # 1. 获取观测
        cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32) 
        current_frame = get_single_frame_obs(data, cmd_vel, last_action_isaac)
        
        # 2. 更新历史缓冲区
        history_buffer = np.roll(history_buffer, -45)
        history_buffer[-45:] = current_frame
        
        # 3. 推理
        ort_inputs = {session.get_inputs()[0].name: history_buffer.reshape(1, -1)}
        action_isaac = session.run(None, ort_inputs)[0].flatten()
        last_action_isaac = action_isaac.copy()
        
        # 4. 计算目标位置并下发
        target_pos_isaac = action_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC
        data.ctrl[:] = target_pos_isaac[isaac_to_mj]
        
        # 5. 执行物理步进
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        # 6. 同步可视化界面
        viewer.sync()
        
        # 7. 调试信息（每50步打印一次）
        step_count += 1
        if step_count % 50 == 0:
            base_height = data.qpos[2]
            print(f"步数: {step_count}, 基座高度: {base_height:.3f}m, 动作范围: [{action_isaac.min():.2f}, {action_isaac.max():.2f}]")
        
        # 8. 精确频率控制
        time_spent = time.time() - loop_start_time
        time_until_next_step = control_dt - time_spent
        
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        else:
            print(f"警告: 控制周期超时 {-time_until_next_step*1000:.1f}ms")