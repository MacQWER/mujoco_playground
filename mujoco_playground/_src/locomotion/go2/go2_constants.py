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

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

# go2 根 body 名称
ROOT_BODY = "base"

# IMU / base-related sensors（名称与 go2 sensor XML 对齐）
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Anchor Policy 相关常量
ANCHOR_OBS_DIM = 40
ANCHOR_ACT_DIM = 12
ANCHOR_PATH = '/data/mujoco_playground/mujoco_playground/experimental/learning/checkpoints/Go2Trot-20260308-090137/trotting_apg_1_5_hz_policy.pkl'

RESIDUAL_OBS_DIM = 48
RESIDUAL_ACT_DIM = 12

# pattern generator 相关常量
STEP_K = 17
GAIT_SCALE = 0.3