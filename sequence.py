import mujoco
model = mujoco.MjModel.from_xml_path("go2.xml")

print("=== 关节顺序 (qpos) ===")
for i in range(model.njnt):
    jnt = model.joint(i)
    print(f"  Joint {i}: {jnt.name}")

print("\n=== 执行器顺序 (ctrl) ===")
for i in range(model.nu):
    act = model.actuator(i)
    print(f"  Actuator {i}: {act.name}")

## 目前已发现的问题：

### 问题 1: 关节顺序不匹配
