from etils import epath

from mujoco_playground._src import mjx_env

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

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

