import mujoco
import mujoco.viewer
import onnxruntime as ort
import numpy as np
import time

# 1. 配置区
MODEL_PATH = "scene2.xml"
POLICY_PATH = "policy.onnx"
ACTION_SCALE = 0.25 # 来自 flat_env_cfg.py

# 严格按照学长的顺序定义初始姿态：FL, FR, RL, RR (Hips -> Thighs -> Calves)
DEFAULT_DOF_POS_ISAAC = np.array([
    0.1, -0.1,  0.1, -0.1,   # Hips
    0.8,  0.8,  1.0,  1.0,   # Thighs
   -1.5, -1.5, -1.5, -1.5    # Calves
], dtype=np.float32)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
session = ort.InferenceSession(POLICY_PATH, providers=['CPUExecutionProvider'])

# 映射索引：MuJoCo默认(FR, FL, RR, RL各腿顺序) -> Isaac学长顺序(按关节类排列)
mj_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_mj = np.argsort(mj_to_isaac)

OBS_DIM = 225 # 45维 * 5步历史
history_buffer = np.zeros(OBS_DIM, dtype=np.float32)
last_action_isaac = np.zeros(12, dtype=np.float32)

def get_single_frame_obs(data, commands, last_action_isaac):
    # A. 角速度 (3)
    ang_vel = data.sensor('imu_gyro').data.copy()
    
    # B. 重力投影 (3) - 修正极性，平站时应为 [0, 0, -1]
    q = data.sensor('imu_quat').data  # [w, x, y, z]
    gravity_proj = np.zeros(3)
    mujoco.mju_rotVecQuat(gravity_proj, np.array([0, 0, 1.0]), q)
    
    # C. 关节位置与速度 (重排为 Isaac 顺序)
    mj_qpos = data.qpos[7:19]
    mj_qvel = data.qvel[6:18]
    
    # 计算相对于默认姿态的偏差
    dof_pos_isaac = mj_qpos[mj_to_isaac] - DEFAULT_DOF_POS_ISAAC
    dof_vel_isaac = mj_qvel[mj_to_isaac]
    
    # D. 拼接当前帧 (45维) - 必须严格遵守 PolicyCfg 顺序
    current_frame = np.concatenate([
        ang_vel,           # 3
        gravity_proj,      # 3
        commands,          # 3
        dof_pos_isaac,     # 12
        dof_vel_isaac,     # 12
        last_action_isaac  # 12
    ])
    return current_frame.astype(np.float32)

# 3. 仿真主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    # --- 初始预设站立环节 (1.5秒钟) ---
    print("正在初始化站立姿态，请等待1.5秒...")
    mj_home_ctrl = DEFAULT_DOF_POS_ISAAC[isaac_to_mj]
    
    # 设置初始物理状态
    data.qpos[7:19] = mj_home_ctrl
    mujoco.mj_forward(model, data) 
    
    # 平稳过渡防止开局炸飞
    for _ in range(500):
        data.ctrl[:] = mj_home_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    
    # --- 缓冲区热启动 ---
    # 使用当前稳定的站立观测填充历史，防止模型因全0历史产生瞬时大动作
    init_obs = get_single_frame_obs(data, np.array([0,0,0]), np.zeros(12))
    for i in range(5):
        history_buffer[i*45 : (i+1)*45] = init_obs

    print("开始行走测试！")
    model.opt.timestep=0.002
    # --- 行走逻辑环节 ---
    control_dt = 0.02  # 50Hz控制频率
    steps_per_control = int(control_dt / model.opt.timestep)
    
    step_count = 0
    target_vel = 0.6  # 稍微提高目标速度观察步态
    current_vel = 0.0
    previous_target = DEFAULT_DOF_POS_ISAAC.copy()

    while viewer.is_running():
        loop_start_time = time.time()
        
        # 1. 指令渐进增长
        if step_count < 150:
            current_vel = target_vel * (step_count / 150.0)
        else:
            current_vel = target_vel
        
        cmd_vel = np.array([current_vel, 0.0, 0.0], dtype=np.float32)
        
        # 2. 获取并同步观测
        current_frame = get_single_frame_obs(data, cmd_vel, last_action_isaac)
        
        if step_count % 50 == 0:
            print("-" * 30)
            print(f"[DEBUG] 重力投影 (预期[0,0,-1]): {current_frame[3:6]}")
            print(f"[DEBUG] 关节位置偏差 (预期接近0): {np.mean(np.abs(current_frame[9:21])):.4f}")

        # 更新历史缓冲区 (45 * 5)
        history_buffer = np.roll(history_buffer, -45)
        history_buffer[-45:] = current_frame
        
        # 3. 推理
        ort_inputs = {session.get_inputs()[0].name: history_buffer.reshape(1, -1)}
        action_isaac = session.run(None, ort_inputs)[0].flatten()
        last_action_isaac = action_isaac.copy()
        
        # 4. 动作映射与平滑
        target_pos_isaac = action_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC
        
        # 动作平滑系数，减缓MuJoCo中的高频冲击
        alpha = 0.5 if step_count < 100 else 0.3
        target_pos_isaac = alpha * previous_target + (1 - alpha) * target_pos_isaac
        previous_target = target_pos_isaac.copy()
        
        # 转换回MuJoCo关节顺序并下发
        data.ctrl[:] = target_pos_isaac[isaac_to_mj]
        
        # 5. 物理仿真步进
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        viewer.sync()
        step_count += 1
        
        # 6. 频率控制
        time_spent = time.time() - loop_start_time
        if control_dt > time_spent:
            time.sleep(control_dt - time_spent)