import mujoco
import mujoco.viewer
import onnxruntime as ort
import numpy as np
import time

# 1. 配置区
MODEL_PATH = "scene.xml"
POLICY_PATH = "policy.onnx"
ACTION_SCALE = 0.15

DEFAULT_DOF_POS_ISAAC = np.array([
    -0.1, 0.1, -0.1, 0.1,   # Hips (FL, FR, RL, RR)
    0.8,  0.8,  0.8,  0.8,   # Thighs (FL, FR, RL, RR)
   -1.3, -1.3, -1.3, -1.3    # Calves (FL, FR, RL, RR)
])
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
session = ort.InferenceSession(POLICY_PATH, providers=['CPUExecutionProvider'])

# 映射索引: 将 MuJoCo 索引重排为 Isaac 训练顺序的映射
mj_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
# 逆映射: 将 Isaac 顺序重排回 MuJoCo 顺序 (用于 ctrl 下发)
isaac_to_mj = np.argsort(mj_to_isaac)

OBS_DIM = 225
history_buffer = np.zeros(OBS_DIM, dtype=np.float32)
last_action_isaac = np.zeros(12, dtype=np.float32)

def get_single_frame_obs(data, commands, last_action_isaac):
    # A. 角速度 (3)
    ang_vel = data.sensor('imu_gyro').data
    
    # B. 重力投影 (3)
    q = data.sensor('imu_quat').data  # [w, x, y, z]
    gravity_proj = np.zeros(3)
    # MuJoCo 内置函数计算重力投影
    mujoco.mju_rotVecQuat(gravity_proj, np.array([0, 0, -1.0]), q)
    
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
    # --- 初始预设站立环节 (1秒钟) ---
    print("正在初始化站立姿态，请等待1秒...")
    # 将 Isaac 顺序的默认姿态映射回 MuJoCo 执行器顺序
    mj_home_ctrl = DEFAULT_DOF_POS_ISAAC[isaac_to_mj]
    
    # 设置初始关节位置，防止开局掉落冲击
    data.qpos[7:19] = mj_home_ctrl
    
    # 仿真步长通常为 0.002s，运行 500 步约等于 1 秒
    for _ in range(500):
        data.ctrl[:] = mj_home_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    
    print("开始行走测试！")
    # --- 行走逻辑环节 ---
    control_dt = 0.01  
    steps_per_control = 2
    
    print(f"开始行走测试！控制频率: {1/control_dt}Hz, 物理步数/周: {steps_per_control}")

    while viewer.is_running():
        loop_start_time = time.time() # 记录循环开始时间
        
        # 1. 获取观测
        # 使用当前设置的指令，你可以根据需要修改 cmd_vel
        cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32) 
        current_frame = get_single_frame_obs(data, cmd_vel, last_action_isaac)
        
        # 2. 更新历史缓冲区 (45 * 5 = 225)
        history_buffer = np.roll(history_buffer, -45)
        history_buffer[-45:] = current_frame
        
        # 3. 推理
        ort_inputs = {session.get_inputs()[0].name: history_buffer.reshape(1, -1)}
        action_isaac = session.run(None, ort_inputs)[0].flatten()
        last_action_isaac = action_isaac.copy()
        
        # 4. 计算目标位置并下发
        # 这里的 ACTION_SCALE 建议根据你之前的反馈微调 (0.15 - 0.25)
        target_pos_isaac = action_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC
        data.ctrl[:] = target_pos_isaac[isaac_to_mj]
        
        # 5. 执行物理步进
        # 确保在一个控制周期内，物理引擎运行了足够的时间
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        # 6. 同步可视化界面
        viewer.sync()
        
        # 7. 精确频率控制
        # 计算本次循环已消耗的时间（推理 + 物理步进）
        time_spent = time.time() - loop_start_time
        time_until_next_step = control_dt - time_spent
        
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # 如果 time_spent > control_dt，说明计算开销太大，无法维持 50Hz