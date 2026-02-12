from typing import List
import threading
from datetime import datetime
from queue import Queue
import json
from dataclasses import dataclass
from copy import deepcopy


import numpy as np
from loop_rate_limiters import RateLimiter
import zenoh
from loguru import logger

from .xr_client import XrClient
from .utils import convert_pose_to_T, callculate_delta_T
from .common import TeleopState, TeleopRecordState, TeleopCommand


class PicoServer:
    def __init__(self, config: dict):
        self.config = deepcopy(config)
        logger.info(f"Pico Server Config: {self.config}")

        self.xr_client = XrClient()
        self.pico_data_queue = Queue(maxsize=50)

        config = zenoh.Config()
        self.session = zenoh.open(config)
        self.pub = self.session.declare_publisher("teleop/vr/pico_data")

        # State
        self.cur_teleop_state = TeleopState.DISABLE
        self.cur_teleop_record_state = TeleopRecordState.NONE
        self.cur_teleop_command = TeleopCommand.NONE
        self.enbale_time_sum = 0.0
        self.enable_time_threshold = 1

        self.pre_pico_data = None
        self.enable_pico_data = None

    def start(self):
        # Publish data via Zenoh
        self.publish_data_thread = threading.Thread(target=self.publish_data)
        self.publish_data_thread.start()

        # Read data from Pico
        self.read_data_thread = threading.Thread(target=self.update)
        self.read_data_thread.start()

        logger.info(f"-------------------Start Pico Server-------------------")

    def _read_data(self):
        pico_data = {
            "raw": {},
            "teleop": {
                "teleop_state": None,
                "teleop_record_state": None,
                "teleop_commands": None,
            },
        }
        timestamp = datetime.now()
        pico_data["timestamp"] = timestamp
        # Pose: [x, y, z, qx, qy, qz, qw]
        pico_data["raw"]["headset"] = self.xr_client.get_pose_by_name("headset")

        # ------
        # Controller pose
        # ------
        pico_data["raw"]["left_controller"] = self.xr_client.get_pose_by_name(
            "left_controller"
        )
        pico_data["raw"]["right_controller"] = self.xr_client.get_pose_by_name(
            "right_controller"
        )

        # ------
        # Trigger
        # ------
        pico_data["raw"]["left_trigger"] = self.xr_client.get_key_value_by_name(
            "left_trigger"
        )
        pico_data["raw"]["right_trigger"] = self.xr_client.get_key_value_by_name(
            "right_trigger"
        )

        # ------
        # Grip
        # ------
        pico_data["raw"]["left_grip"] = self.xr_client.get_key_value_by_name(
            "left_grip"
        )
        pico_data["raw"]["right_grip"] = self.xr_client.get_key_value_by_name(
            "right_grip"
        )

        # ------
        # A/B/X/Y
        # ------
        pico_data["raw"]["A"] = self.xr_client.get_button_state_by_name("A")
        pico_data["raw"]["B"] = self.xr_client.get_button_state_by_name("B")
        pico_data["raw"]["X"] = self.xr_client.get_button_state_by_name("X")
        pico_data["raw"]["Y"] = self.xr_client.get_button_state_by_name("Y")

        # ------
        # Menu button
        # ------
        pico_data["raw"]["left_menu_button"] = self.xr_client.get_button_state_by_name(
            "left_menu_button"
        )

        # ------
        # Left/Right axis click
        # ------
        pico_data["raw"]["left_axis"] = self.xr_client.get_joystick_state("left")
        pico_data["raw"]["right_axis"] = self.xr_client.get_joystick_state("right")

        return pico_data

    def _update_teleop_state(self, pico_data):
        if self.pre_pico_data is None:
            return TeleopState.DISABLE
        if self.cur_teleop_state == TeleopState.DISABLE:
            if (
                self.pre_pico_data["raw"]["left_grip"] > 0.5
                and self.pre_pico_data["raw"]["right_grip"] > 0.5
            ) and (
                pico_data["raw"]["left_grip"] > 0.5
                and pico_data["raw"]["right_grip"] > 0.5
            ):
                timestamp_gap = (
                    pico_data["timestamp"] - self.pre_pico_data["timestamp"]
                ).total_seconds()
                self.enbale_time_sum += timestamp_gap
                logger.info(f"Enable time sum: {self.enbale_time_sum}")
                if self.enbale_time_sum >= self.enable_time_threshold:
                    self.enable_pico_data = deepcopy(pico_data)
                    logger.info(f"-------------------Enable Teleop-------------------")
                    return TeleopState.ENABLE
                else:
                    return TeleopState.DISABLE
            else:
                return TeleopState.DISABLE
        else:
            # ------
            # 禁用
            # ------
            if (
                self.pre_pico_data["raw"]["left_menu_button"] == 0
                and pico_data["raw"]["left_menu_button"] == 1
            ):
                logger.info(f"-------------------Disable Teleop-------------------")
                self.reset()
                return TeleopState.DISABLE

            return TeleopState.ENABLE

    def _update_teleop_record_state(self, pico_data):
        if self.cur_teleop_state != TeleopState.ENABLE:
            return TeleopRecordState.NONE
        # ------
        # 开始录制
        # ------
        if self.pre_pico_data["raw"]["A"] == 0 and pico_data["raw"]["A"] == 1:
            return TeleopRecordState.START_RECORD
        # ------
        # 结束录制（默认保存）
        # ------
        elif (pico_data["raw"]["right_grip"] == 0) and (
            self.pre_pico_data["raw"]["B"] == 0 and pico_data["raw"]["B"] == 1
        ):
            return TeleopRecordState.END_RECORD
        # ------
        # 删除上一条
        # ------
        # elif (self.pre_pico_data["right_grip"] == 0 and self.pre_pico_data["B"] == 0) and \
        #         (pico_data["right_grip"] == 1 and pico_data["B"] == 1):
        elif pico_data["raw"]["right_grip"] == 1 and (
            self.pre_pico_data["raw"]["B"] == 0 and pico_data["raw"]["B"] == 1
        ):
            return TeleopRecordState.DELET_PRE_RECORD
        else:
            return TeleopRecordState.NONE

    def _update_teleop_commands(self, pico_data) -> List:
        if self.cur_teleop_state == TeleopState.DISABLE:
            return []
        # ------
        # 重置
        # ------
        if self.pre_pico_data["raw"]["X"] == 0 and pico_data["raw"]["X"] == 1:
            return [TeleopCommand.RESET]

        commands = []
        # ------
        # 左执行器触发
        # ------
        if (
            self.pre_pico_data["raw"]["left_trigger"] < 1
            and pico_data["raw"]["left_trigger"] == 1
        ):
            commands.append(TeleopCommand.LEFT_EE_TRIGGER)
        # ------
        # 右执行器触发
        # ------
        if (
            self.pre_pico_data["raw"]["right_trigger"] < 1
            and pico_data["raw"]["right_trigger"] == 1
        ):
            commands.append(TeleopCommand.RIGHT_EE_TRIGGER)

        return commands

    def _calculate_controller_delta_T(self, pico_data):
        head_T = convert_pose_to_T(pico_data["raw"]["headset"])
        head_T_inv = np.linalg.inv(head_T)
        head_init_T = convert_pose_to_T(self.enable_pico_data["raw"]["headset"])
        head_init_T_inv = np.linalg.inv(head_init_T)
        left_init_T = head_init_T_inv @ convert_pose_to_T(
            self.enable_pico_data["raw"]["left_controller"]
        )
        right_init_T = head_init_T_inv @ convert_pose_to_T(
            self.enable_pico_data["raw"]["right_controller"]
        )

        left_T = head_T_inv @ convert_pose_to_T(pico_data["raw"]["left_controller"])
        left_controller_delta_T = np.linalg.inv(left_init_T) @ left_T
        # pico_data["teleop"]["left_controller_delta_T"] = left_controller_delta_T

        right_T = head_T_inv @ convert_pose_to_T(pico_data["raw"]["right_controller"])
        right_controller_delta_T = np.linalg.inv(right_init_T) @ right_T
        # pico_data["teleop"]["right_controller_delta_T"] = right_controller_delta_T

        return left_controller_delta_T, right_controller_delta_T

    def update(self):
        # print(self.pico_data_queue.full())
        rate_limiter = RateLimiter(frequency=self.config["frequency"], warn=False)
        while True:
            # Read pico raw data
            pico_data = self._read_data()

            # ------
            # Update
            # ------
            # Update teleop state
            self.cur_teleop_state = self._update_teleop_state(pico_data)
            pico_data["teleop"]["teleop_state"] = self.cur_teleop_state
            if self.cur_teleop_state == TeleopState.ENABLE:
                # Update teleop record state
                self.cur_teleop_record_state = self._update_teleop_record_state(
                    pico_data
                )
                pico_data["teleop"][
                    "teleop_record_state"
                ] = self.cur_teleop_record_state
                # Update teleop command
                pico_data["teleop"]["teleop_commands"] = self._update_teleop_commands(
                    pico_data
                )
                # Update controller delta T
                left_controller_delta_T, right_controller_delta_T = (
                    self._calculate_controller_delta_T(pico_data)
                )
                pico_data["teleop"]["left_controller_delta_T"] = left_controller_delta_T
                pico_data["teleop"][
                    "right_controller_delta_T"
                ] = right_controller_delta_T
            else:
                pico_data["teleop"]["teleop_record_state"] = TeleopRecordState.NONE
                pico_data["teleop"]["teleop_commands"] = []

            if (
                len(pico_data["teleop"]["teleop_commands"])
                and pico_data["teleop"]["teleop_commands"][0] == TeleopCommand.RESET
            ):
                self.reset()

            # Update queue
            self.pico_data_queue.put(pico_data)
            self.pre_pico_data = deepcopy(pico_data)

            # Log
            if self.cur_teleop_record_state != TeleopRecordState.NONE:
                logger.info(f"teleop_record_state: {self.cur_teleop_record_state.name}")
            if pico_data["teleop"]["teleop_commands"]:
                logger.info(
                    f"teleop_commands: {[x.name for x in pico_data['teleop']['teleop_commands']]}"
                )

            rate_limiter.sleep()

    def reset(self):
        self.enbale_time_sum = 0.0
        self.cur_teleop_state = TeleopState.DISABLE
        self.cur_teleop_command = TeleopCommand.NONE
        self.cur_teleop_record_state = TeleopRecordState.NONE

    def publish_data(self):
        rate_limiter = RateLimiter(frequency=self.config["frequency"], warn=True)
        while True:
            rate_limiter.sleep()
            if self.pico_data_queue.empty():
                # rate_limiter.sleep()
                continue
            pico_data = self.pico_data_queue.get()
            # ------
            # Preprocess for json dumps
            # ------
            pico_data["timestamp"] = pico_data["timestamp"].timestamp()
            if pico_data["teleop"].get("left_controller_delta_T") is not None:
                pico_data["teleop"]["left_controller_delta_T"] = pico_data["teleop"][
                    "left_controller_delta_T"
                ].tolist()
            if pico_data["teleop"].get("right_controller_delta_T") is not None:
                pico_data["teleop"]["right_controller_delta_T"] = pico_data["teleop"][
                    "right_controller_delta_T"
                ].tolist()

            self.pub.put(json.dumps(pico_data))
