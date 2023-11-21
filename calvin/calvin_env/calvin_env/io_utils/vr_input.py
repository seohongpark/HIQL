import collections
import logging

import numpy as np
import pybullet as p
import quaternion  # noqa

import calvin_env.utils.utils as utils

# A logger for this file
log = logging.getLogger(__name__)


# identical for both controllers
RELEVANT_BUTTONS = {
    "button_a": 7,
    "button_b": 1,
    "thumb_trigger": 2,  # on the left of the (right) controller; also 34 sent
    "index_trigger": 33,
}


class ButtonState(collections.namedtuple("ButtonState", ("button_name", "is_down", "was_triggered", "was_released"))):
    __slots__ = ()

    @classmethod
    def from_flag(cls, button_name, flag):
        return cls(
            button_name,
            bool(flag & p.VR_BUTTON_IS_DOWN),
            bool(flag & p.VR_BUTTON_WAS_TRIGGERED),
            bool(flag & p.VR_BUTTON_WAS_RELEASED),
        )


class VrEvent(
    collections.namedtuple(
        "VrEvent",
        [
            "controller_id",
            "position",
            "orientation",
            "analog_axis",
            "n_button_events",
            "n_move_events",
            "buttons",
            "device_type_flag",
        ],
    )
):
    __slots__ = ()

    @property
    def button_dicts(self):
        return {
            button_name: ButtonState.from_flag(button_name, self.buttons[button_idx])
            for button_name, button_idx in RELEVANT_BUTTONS.items()
        }
        # overall 64 buttons according to OpenVR
        # enumerate([f'button_{i}' for i in range(64)])}

    @property
    def device_type(self):
        return {p.VR_DEVICE_HMD: "hmd", p.VR_DEVICE_CONTROLLER: "controller", p.VR_DEVICE_GENERIC_TRACKER: "generic"}[
            self.device_type_flag
        ]


class VrInput:
    """
    pyBullet VR Data Collector
    """

    def __init__(self, vr_controller, limit_angle, visualize_vr_pos, reset_button_queue_len):
        self.vr_controller_id = vr_controller.vr_controller_id
        self.vr_controller = vr_controller
        self.POSITION = vr_controller.POSITION
        self.ORIENTATION = vr_controller.ORIENTATION
        self.ANALOG = vr_controller.ANALOG
        self.BUTTONS = vr_controller.BUTTONS
        self.BUTTON_A = vr_controller.BUTTON_A
        self.BUTTON_B = vr_controller.BUTTON_B
        if limit_angle is not None:
            self.limit_angle = limit_angle[0]
            self.limit_vector = limit_angle[1:]
        else:
            self.limit_angle = None
        self.gripper_position_offset = np.array(vr_controller.gripper_position_offset)
        self.gripper_orientation_offset = quaternion.from_euler_angles(vr_controller.gripper_orientation_offset)
        self.visualize_vr_pos = visualize_vr_pos
        self.vr_pos_uid = None
        if visualize_vr_pos:
            self.vr_pos_uid = self.create_vr_pos_visualization_shape()
        log.info("Disable Picking")
        p.configureDebugVisualizer(p.COV_ENABLE_VR_PICKING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        self._prev_vr_events = None
        self.prev_action = None
        self._reset_button_pressed = False
        self._start_button_pressed = False
        self._prev_reset_button_pressed = False
        self._prev_start_button_pressed = False
        self.reset_button_press_counter = 0
        self.reset_button_queue_len = reset_button_queue_len
        # wait until first vr action event arrives
        while self.prev_action is None:
            _ = self.get_vr_action()

    def create_vr_pos_visualization_shape(self):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=0.005)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=[0, 0, 0])

    def get_vr_action(self):
        # self._reset_button_pressed = False
        self._prev_reset_button_pressed = self._reset_button_pressed
        self._prev_start_button_pressed = self._start_button_pressed
        vr_events = p.getVREvents()
        if vr_events != ():
            self._prev_vr_events = vr_events
            for event in vr_events:
                # Only use one controller
                # if event[0] == self.vr_controller_id:
                action = (event[self.POSITION], event[self.ORIENTATION], event[self.ANALOG])

                self._reset_button_pressed = event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_IS_DOWN > 0
                self._start_button_pressed = event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_IS_DOWN > 0
                robot_action = self.vr_action_to_robot_action(action)

                self.update_reset_button_queue()
                self.prev_action = robot_action
                return robot_action

        return self.prev_action

    def vr_action_to_robot_action(self, action):
        controller_pos, vr_controller_orientation, controller_analogue_axis = action
        desired_ee_pos = controller_pos + self.gripper_position_offset
        orientation = utils.xyzw_to_wxyz(vr_controller_orientation)
        q1 = quaternion.from_float_array(orientation)
        q2 = self.gripper_orientation_offset
        q12 = q1 * q2
        arr = quaternion.as_float_array(q12)
        desired_ee_orn = utils.wxyz_to_xyzw(arr)

        # v2 = quaternion.rotate_vectors(q12, [0, 0, 1])
        # if self.limit_angle is not None:
        #     if utils.angle_between(self.limit_vector, v2) <= self.limit_angle / 180 * math.pi:
        #         self.prev_ee_orn = desired_ee_orn
        #     else:
        #         desired_ee_orn = self.prev_ee_orn

        if controller_analogue_axis > 0.1:
            gripper_action = -1
        else:
            gripper_action = 1

        # change color of vr pos sphere when starting or ending recording
        if self.visualize_vr_pos:
            if self.start_button_pressed:
                p.changeVisualShape(self.vr_pos_uid, -1, rgbaColor=[0, 1, 0, 1])
            if self.reset_button_pressed:
                p.changeVisualShape(self.vr_pos_uid, -1, rgbaColor=[1, 0, 0, 1])
            p.resetBasePositionAndOrientation(self.vr_pos_uid, desired_ee_pos, desired_ee_orn)

        return desired_ee_pos, desired_ee_orn, gripper_action

    def update_reset_button_queue(self):
        if self._reset_button_pressed:
            self.reset_button_press_counter += 1
        else:
            self.reset_button_press_counter = 0

    @property
    def reset_button_hold(self):
        return self.reset_button_press_counter >= self.reset_button_queue_len

    @property
    def reset_button_pressed(self):
        return self._reset_button_pressed and not self._prev_reset_button_pressed

    @property
    def start_button_pressed(self):
        return self._start_button_pressed and not self._prev_start_button_pressed

    @property
    def prev_vr_events(self):
        return self._prev_vr_events
