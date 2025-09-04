# visualize_drone_mujoco.py
import mujoco
from mujoco.viewer import launch

# load your XML file
model = mujoco.MjModel.from_xml_path("examples/bitcraze_crazyflie_2/cf2.xml")
data  = mujoco.MjData(model)

# launch interactive viewer
viewer = launch(model, data)
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.render()
