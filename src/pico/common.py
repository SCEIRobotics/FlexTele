from enum import IntEnum


class TeleopState(IntEnum):
    DISABLE = -1  # 禁用
    ENABLE = 0  # 启用


class TeleopRecordState(IntEnum):
    NONE = -1
    START_RECORD = 0  # 开始录制
    END_RECORD = 1  # 结束录制
    DELET_PRE_RECORD = 2  # 删除前一次录制 or SAVE_RECORD


class TeleopCommand(IntEnum):
    NONE = -1
    RESET = 0  # 重置
    LEFT_EE_TRIGGER = 1  # 左末端触发
    RIGHT_EE_TRIGGER = 2  # 右末端触发
