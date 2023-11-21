#!/usr/bin/python3
from copy import deepcopy
import logging
import os
import sys

import hydra
import pybullet as p
import quaternion  # noqa

from calvin_env.io_utils.data_recorder import DataRecorder
from calvin_env.io_utils.vr_input import VrInput

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_data_collection")
def main(cfg):
    # Load Scene
    env = hydra.utils.instantiate(cfg.env)
    vr_input = hydra.utils.instantiate(cfg.vr_input)

    data_recorder = None
    if cfg.recorder.record:
        data_recorder = DataRecorder(env, cfg.recorder.record_fps, cfg.recorder.enable_tts)

    log.info("Initialization done!")
    log.info("Entering Loop")

    record = False

    while 1:
        # get input events
        action = vr_input.get_vr_action()
        obs, _, _, info = env.step(action)
        done = False
        if vr_input.reset_button_pressed:
            done = True
        if vr_input.start_button_pressed:
            record = True
        if vr_input.reset_button_hold:
            data_recorder.delete_episode()
        if record and cfg.recorder.record:
            data_recorder.step(vr_input.prev_vr_events, obs, done, info)
        if done:
            record = False
            env.reset()


if __name__ == "__main__":
    main()
