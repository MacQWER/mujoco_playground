from etils import epath

from mujoco_playground._src import mjx_env

FEET_GEOMS = ["FL", "FR", "RL", "RR"]

HIP_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2"

MJX_XML_PATH = (
    ROOT_PATH / "xmls" / "scene_mjx_collision_free.xml"
)

MUJOCO_XML_PATH = (
    ROOT_PATH / "xmls" / "scene.xml"
)

MJX_XML_SENSOR_PATH = (
    ROOT_PATH / "xmls" / "scene_mjx_collision_free_sensor.xml"
)

ONNX_DIR = mjx_env.ROOT_PATH / "experimental" / "sim2sim" / "onnx"

ROOT_BODY = "base"

# go2 feet sites（注意：go2 使用 *_foot）
FEET_SITES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

FEET_POS_SENSOR = [f"{geom}_pos" for geom in FEET_GEOMS]

# go2 根 body 名称
ROOT_BODY = "base"

# IMU / base-related sensors（名称与 go2 sensor XML 对齐）
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Observation layout / scaling.
OBS_W_LOCAL_SCALE = 0.25
OBS_JOINT_VELS_SCALE = 0.05

OBS_W_LOCAL_DIM = 3
OBS_G_LOCAL_DIM = 3
OBS_COMMAND_DIM = 3
OBS_ANGLES_DIM = 12
OBS_JOINT_VELS_DIM = 12
OBS_LAST_ACTION_DIM = 12
OBS_KIN_REF_DIM = 12
OBS_ANCHOR_ACTION_DIM = 12

OBS_W_LOCAL_SLICE = slice(0, 3)
OBS_G_LOCAL_SLICE = slice(3, 6)
OBS_COMMAND_SLICE = slice(6, 9)
OBS_ANGLES_SLICE = slice(9, 21)
OBS_JOINT_VELS_SLICE = slice(21, 33)
OBS_LAST_ACTION_SLICE = slice(33, 45)
OBS_KIN_REF_SLICE = slice(45, 57)
OBS_ANCHOR_ACTION_SLICE = slice(57, 69)

# Anchor Policy 相关常量
ANCHOR_OBS_DIM = OBS_ANCHOR_ACTION_SLICE.stop
ANCHOR_ACT_DIM = 12
ANCHOR_PATH = None
# ANCHOR_PATH = '/data/mujoco_playground/mujoco_playground/experimental/learning/checkpoints/Go2Trot-20260320-100219/trotting_apg_2_hz_policy.pkl'

RESIDUAL_OBS_DIM = ANCHOR_OBS_DIM
RESIDUAL_ACT_DIM = 12
# RESIDUAL_PATH = '/data/mujoco_playground/mujoco_playground/experimental/learning/checkpoints/Go2Joystick2-20260324-080452-apg/params.pkl'
RESIDUAL_PATH = '/data/mujoco_playground/mujoco_playground/experimental/learning/checkpoints/Go2Joystick2-20260403-070849-apg/params.pkl'

# pattern generator 相关常量
STEP_K = 13
GAIT_SCALE = 0.3