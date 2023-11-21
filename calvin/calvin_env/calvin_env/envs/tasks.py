from functools import partial
from typing import Dict, Set

import numpy as np
from omegaconf import ListConfig
from scipy.spatial.transform import Rotation as R


class Tasks:
    def __init__(self, tasks):
        """
        A task is defined as a specific change between the start_info and end_info dictionaries.
        Use config file in conf/tasks/ to define tasks using the base task functions defined in this class
        """
        # register task functions from config file
        self.tasks = {name: partial(getattr(self, args[0]), *args[1:]) for name, args in dict(tasks).items()}
        # dictionary mapping from task name to task id
        self.task_to_id = {name: i for i, name in enumerate(self.tasks.keys())}
        # dictionary mapping from task id to task name
        self.id_to_task = {i: name for i, name in enumerate(self.tasks.keys())}

    def get_task_info(self, start_info: Dict, end_info: Dict) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if function(start_info=start_info, end_info=end_info)
        }

    def get_task_info_for_set(self, start_info: Dict, end_info: Dict, task_filter: Set) -> Set:
        """
        start_info: dict with scene info and robot info
        end_info: dict with scene info and robot info
        task_filter: set with task names to check
        returns set with achieved tasks
        """
        # call functions that are registered in self.tasks
        return {
            task_name
            for task_name, function in self.tasks.items()
            if task_name in task_filter and function(start_info=start_info, end_info=end_info)
        }

    @property
    def num_tasks(self):
        return len(self.tasks)

    @staticmethod
    def rotate_object(
        obj_name, z_degrees, x_y_threshold=30, z_threshold=180, movement_threshold=0.1, start_info=None, end_info=None
    ):
        """
        Returns True if the object with obj_name was rotated more than z_degrees degrees around the z-axis while not
        being rotated more than x_y_threshold degrees around the x or y axis.
        z_degrees is negative for clockwise rotations and positive for counter-clockwise rotations.
        """
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
        start_orn = R.from_quat(obj_start_info["current_orn"])
        end_orn = R.from_quat(obj_end_info["current_orn"])
        rotation = end_orn * start_orn.inv()
        x, y, z = rotation.as_euler("xyz", degrees=True)

        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        if np.linalg.norm(pos_diff) > movement_threshold:
            return False

        end_contacts = set(c[2] for c in obj_end_info["contacts"])
        robot_uid = {start_info["robot_info"]["uid"]}
        if len(end_contacts - robot_uid) == 0:
            return False

        if z_degrees > 0:
            return z_degrees < z < z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold
        else:
            return z_degrees > z > -z_threshold and abs(x) < x_y_threshold and abs(y) < x_y_threshold

    @staticmethod
    def push_object(obj_name, x_direction, y_direction, start_info, end_info):
        """
        Returns True if the object with 'obj_name' was moved more than 'x_direction' meters in x direction
        (or 'y_direction' meters in y direction analogously).
        Note that currently x and y pushes are mutually exclusive, meaning that one of the arguments has to be 0.
        The sign matters, e.g. pushing an object to the right when facing the table coincides with a movement in
        positive x-direction.
        """
        assert x_direction * y_direction == 0 and x_direction + y_direction != 0
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos

        robot_uid = start_info["robot_info"]["uid"]
        # contacts excluding robot
        start_contacts = set((c[2], c[4]) for c in obj_start_info["contacts"] if c[2] != robot_uid)
        end_contacts = set((c[2], c[4]) for c in obj_end_info["contacts"] if c[2] != robot_uid)

        # computing set difference to check if object had surface contact (excluding robot) at both times
        surface_contact = len(start_contacts) > 0 and len(end_contacts) > 0 and start_contacts <= end_contacts
        if not surface_contact:
            return False

        if x_direction > 0:
            return pos_diff[0] > x_direction
        elif x_direction < 0:
            return pos_diff[0] < x_direction

        if y_direction > 0:
            return pos_diff[1] > y_direction
        elif y_direction < 0:
            return pos_diff[1] < y_direction

    @staticmethod
    def lift_object(obj_name, z_direction, surface_body=None, surface_link=None, start_info=None, end_info=None):
        """
        Returns True if the object with 'obj_name' was grasped by the robot and lifted more than 'z_direction' meters.
        """
        assert z_direction > 0
        obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
        obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]

        start_pos = np.array(obj_start_info["current_pos"])
        end_pos = np.array(obj_end_info["current_pos"])
        pos_diff = end_pos - start_pos
        z_diff = pos_diff[2]

        robot_uid = start_info["robot_info"]["uid"]
        start_contacts = set(c[2] for c in obj_start_info["contacts"])
        end_contacts = set(c[2] for c in obj_end_info["contacts"])

        surface_criterion = True
        if surface_body and surface_link is None:
            surface_uid = start_info["scene_info"]["fixed_objects"][surface_body]["uid"]
            surface_criterion = surface_uid in start_contacts
        elif surface_body and surface_link:
            surface_uid = start_info["scene_info"]["fixed_objects"][surface_body]["uid"]
            surface_link_id = start_info["scene_info"]["fixed_objects"][surface_body]["links"][surface_link]
            start_contacts_links = set((c[2], c[4]) for c in obj_start_info["contacts"])
            surface_criterion = (surface_uid, surface_link_id) in start_contacts_links

        return (
            z_diff > z_direction
            # and robot_uid not in start_contacts
            and robot_uid in end_contacts
            and len(end_contacts) == 1
            and surface_criterion
        )

    @staticmethod
    def place_object(dest_body, dest_link=None, start_info=None, end_info=None):
        """
        Returns True if the object that the robot has currently lifted is placed on the body 'dest_body'
        (on 'dest_link' if provided).
        The robot may not touch the object after placing.
        """
        robot_uid = start_info["robot_info"]["uid"]

        robot_contacts_start = set(c[2] for c in start_info["robot_info"]["contacts"])
        robot_contacts_end = set(c[2] for c in end_info["robot_info"]["contacts"])
        if not len(robot_contacts_start) == 1:
            return False
        obj_uid = list(robot_contacts_start)[0]

        if obj_uid in robot_contacts_end:
            return False
        _obj_name = [k for k, v in start_info["scene_info"]["movable_objects"].items() if v["uid"] == obj_uid]
        if not len(_obj_name) == 1:
            return False
        obj_name = _obj_name[0]

        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]

        object_contacts_start = set(c[2] for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        if dest_link is None:
            object_contacts_end = set(c[2] for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and dest_uid in object_contacts_end
            )
        else:
            dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]
            end_contacts_links = set(
                (c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"]
            )
            return (
                robot_uid in object_contacts_start
                and len(object_contacts_start) == 1
                and (dest_uid, dest_link_id) in end_contacts_links
            )

    @staticmethod
    def push_object_into(obj_name, src_body, src_link, dest_body, dest_link, start_info=None, end_info=None):
        """
        obj_name is either a list of object names or a string
        Returns True if the object / any of the objects changes contact from src_body to dest_body.
        The robot may neither touch the object at start nor end.
        """
        if isinstance(obj_name, (list, ListConfig)):
            return any(
                Tasks.push_object_into(ob, src_body, src_link, dest_body, dest_link, start_info, end_info)
                for ob in obj_name
            )
        robot_uid = start_info["robot_info"]["uid"]

        src_uid = start_info["scene_info"]["fixed_objects"][src_body]["uid"]
        src_link_id = start_info["scene_info"]["fixed_objects"][src_body]["links"][src_link]
        dest_uid = end_info["scene_info"]["fixed_objects"][dest_body]["uid"]
        dest_link_id = end_info["scene_info"]["fixed_objects"][dest_body]["links"][dest_link]

        start_contacts = set((c[2], c[4]) for c in start_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        end_contacts = set((c[2], c[4]) for c in end_info["scene_info"]["movable_objects"][obj_name]["contacts"])
        return (
            robot_uid not in start_contacts | end_contacts
            and len(start_contacts) == 1
            and (src_uid, src_link_id) in start_contacts
            and (dest_uid, dest_link_id) in end_contacts
        )

    @staticmethod
    def move_door_abs(joint_name, start_threshold, end_threshold, start_info, end_info):
        """
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """
        start_joint_state = start_info["scene_info"]["doors"][joint_name]["current_state"][0]
        end_joint_state = end_info["scene_info"]["doors"][joint_name]["current_state"][0]

        if start_threshold < end_threshold:
            return start_joint_state < start_threshold < end_threshold < end_joint_state
        elif start_threshold > end_threshold:
            return start_joint_state > start_threshold > end_threshold > end_joint_state
        else:
            raise ValueError

    @staticmethod
    def move_door_rel(joint_name, threshold, start_info, end_info):
        """
        Returns True if the joint specified by 'obj_name' and 'joint_name' (e.g. a door or drawer)
        is moved from at least 'start_threshold' to 'end_threshold'.
        """
        start_joint_state = start_info["scene_info"]["doors"][joint_name]["current_state"]
        end_joint_state = end_info["scene_info"]["doors"][joint_name]["current_state"]

        return (
            0 < threshold < end_joint_state - start_joint_state or 0 > threshold > end_joint_state - start_joint_state
        )

    @staticmethod
    def toggle_light(light_name, start_state, end_state, start_info, end_info):
        return (
            start_info["scene_info"]["lights"][light_name]["logical_state"] == start_state
            and end_info["scene_info"]["lights"][light_name]["logical_state"] == end_state
        )

    @staticmethod
    def stack_objects(max_vel=1, start_info=None, end_info=None):
        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())

        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                not len(obj_uids & obj_start_contacts)
                and len(obj_uids & obj_end_contacts)
                and not len(obj_end_contacts - obj_uids)
            ):
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_end_info["current_lin_vel"]) < max_vel) and np.all(
                    np.abs(obj_end_info["current_ang_vel"]) < max_vel
                ):
                    return True
        return False

    @staticmethod
    def unstack_objects(max_vel=1, start_info=None, end_info=None):
        obj_uids = set(obj["uid"] for obj in start_info["scene_info"]["movable_objects"].values())

        for obj_name in start_info["scene_info"]["movable_objects"]:
            obj_start_info = start_info["scene_info"]["movable_objects"][obj_name]
            obj_end_info = end_info["scene_info"]["movable_objects"][obj_name]
            obj_start_contacts = set(c[2] for c in obj_start_info["contacts"])
            obj_end_contacts = set(c[2] for c in obj_end_info["contacts"])

            if (
                len(obj_uids & obj_start_contacts)
                and not len(obj_start_contacts - obj_uids)
                and not len(obj_uids & obj_end_contacts)
            ):
                # object velocity may not exceed max_vel for successful stack
                if np.all(np.abs(obj_start_info["current_lin_vel"]) < max_vel) and np.all(
                    np.abs(obj_start_info["current_ang_vel"]) < max_vel
                ):
                    return True
        return False
