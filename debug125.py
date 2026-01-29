import mujoco
import numpy as np

# 1. 加载模型
# 请确保路径与你的项目一致
MODEL_PATH = "scene2.xml"
try:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# 2. 定义学长 unitree.py 中的标准顺序 (Isaac 顺序)
# 来源自 unitree.py 的 joint_sdk_names 字段
isaac_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

# 3. 获取 MuJoCo 中的关节顺序
# MuJoCo 的 qpos[7:19] 对应这 12 个自由度
print("\n--- MuJoCo 关节顺序检查 ---")
mj_joint_names = []
# 遍历所有关节，排除 freejoint (root)
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    # 忽略自由根关节
    if joint_name == "freejoint" or joint_name is None:
        continue
    mj_joint_names.append(joint_name)

for idx, name in enumerate(mj_joint_names):
    print(f"MuJoCo 索引 [{idx}]: {name}")

# 4. 自动生成 mj_to_isaac 映射表
# 逻辑：mj_to_isaac[i] = j 表示 MuJoCo 的第 i 个关节对应 Isaac 列表的第 j 个
print("\n--- 建议的 mj_to_isaac 映射表 ---")
suggested_mapping = []
try:
    for mj_name in mj_joint_names:
        isaac_idx = isaac_joint_names.index(mj_name)
        suggested_mapping.append(isaac_idx)
    
    print(f"mj_to_isaac = {suggested_mapping}")
except ValueError as e:
    print(f"错误: MuJoCo 中的关节名称与 Isaac 列表不匹配！")
    print(f"请检查 XML 中的 joint name 是否严格为 'FR_hip_joint' 等格式。")

# 5. 验证 DEFAULT_DOF_POS_ISAAC
# 对应 unitree.py 中的 init_state
print("\n--- 初始姿态对齐检查 (unitree.py) ---")
# 提取 unitree.py 中的初始值
# FR: Hip -0.1, Thigh 0.8, Calf -1.5
# FL: Hip 0.1, Thigh 0.8, Calf -1.5
# RR: Hip -0.1, Thigh 1.0, Calf -1.5
# RL: Hip 0.1, Thigh 1.0, Calf -1.5
print("请确保你的代码中 DEFAULT_DOF_POS_ISAAC 的值如下（按 Isaac 顺序）:")
print("Hips: [FR:-0.1, FL:0.1, RR:-0.1, RL:0.1] (注意顺序可能因你的数组定义而异)")