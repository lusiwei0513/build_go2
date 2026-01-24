import mujoco
import mujoco.viewer
import onnxruntime as ort
import numpy as np
import time

# 1. 配置区
MODEL_PATH = "scene.xml"  # 请确保该文件在当前目录
POLICY_PATH = "policy.onnx"
ACTION_SCALE = 0.25
# 默认站立姿态，来自 unitree.py
DEFAULT_DOF_POS = np.array([0, 0.7, -1.5, 0, 0.7, -1.5, 0, 0.7, -1.5, 0, 0.7, -1.5])

# 2. 初始化 ONNX 和 MuJoCo
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
# 强制使用 CPU 推理以保证兼容性
session = ort.InferenceSession(POLICY_PATH, providers=['CPUExecutionProvider'])

# 历史缓冲区：225 维 / 假设单帧 45 维 = 5 帧
OBS_DIM = 225
history_buffer = np.zeros(OBS_DIM, dtype=np.float32)
last_action = np.zeros(12, dtype=np.float32)

def get_single_frame_obs(data, commands, last_action):
    # 根据 velocity_env_cfg.py 顺序拼接
    lin_vel = data.sensor('frame_vel').data
    ang_vel = data.sensor('imu_gyro').data
    
    # 计算重力投影向量
    q = data.sensor('imu_quat').data  # [w, x, y, z]
    gravity_proj = np.array([
        2 * (q[1]*q[3] - q[0]*q[2]),
        2 * (q[2]*q[3] + q[0]*q[1]),
        1 - 2 * (q[1]**2 + q[2]**2)
    ])
    
    dof_pos = data.qpos[7:] - DEFAULT_DOF_POS  # 减去基座的7位坐标
    dof_vel = data.qvel[6:]
    
    # 拼接当前帧 (注意：如果训练包含 height_scan，此处需填充全 0)
    # 此处假设单帧组成为：lin_vel(3), ang_vel(3), grav(3), cmd(3), pos(12), vel(12), last_act(12) = 48维
    # 剩余维度需根据实际 config 调整或补 0
    current_frame = np.concatenate([lin_vel, ang_vel, gravity_proj, commands, dof_pos, dof_vel, last_action])
    return current_frame

# 3. 仿真主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("仿真已启动！按空格键开始/暂停。")
    step_start = time.time()
    
    # 在 while 循环前增加：让机器人先用 1 秒钟平稳站立
    for _ in range(200): 
        data.ctrl[:] = DEFAULT_DOF_POS
        mujoco.mj_step(model, data)

    while viewer.is_running():
        # 获取当前帧观测
        commands = np.array([0.6, 0.0, 0.0]) # 设置前进速度指令
        #current_obs = get_single_frame_obs(data, commands, last_action)
        
        # 更新历史缓冲区（左移并填充新帧）
        #history_buffer = np.roll(history_buffer, -len(current_obs))
        #history_buffer[-len(current_obs):] = current_obs
        
        # 推理
        #ort_inputs = {session.get_inputs()[0].name: history_buffer.reshape(1, -1)}
        #action = session.run(None, ort_inputs)[0].flatten()
        
        # 施加控制
        #data.ctrl[:] = action * ACTION_SCALE + DEFAULT_DOF_POS
        #data.ctrl[:] = DEFAULT_DOF_POS
        #last_action = action.copy()
        
        # 物理步进
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 维持实时频率
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        step_start = time.time()