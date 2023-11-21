#!/usr/bin/python3

from collections import deque
import logging
from multiprocessing import Process
import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pybullet as p
from scipy.spatial.transform.rotation import Rotation as R

from calvin_env.utils.utils import count_frames, get_episode_lengths, set_egl_device, to_relative_action

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_rendering")
def main(cfg):
    log.info("pyBullet Data Renderer")
    log.info("Determining maximum frame")

    recording_dir = (Path(hydra.utils.get_original_cwd()) / cfg.load_dir).absolute()
    max_frames = count_frames(recording_dir)
    log.info(f"Found continuous interval of {max_frames} frames without gaps")

    num_prev_rendered_episodes = num_previously_rendered_episodes()
    if num_prev_rendered_episodes == 0:
        playback_cfg = build_rendering_config(recording_dir, rendering_config=cfg)
    else:
        playback_cfg = load_rendering_config(cfg)

    log.info("Initialization done!")
    log.info(f"Starting {cfg.processes} processes")

    if playback_cfg.set_static_cam:
        playback_cfg = set_static_cams_from_gui(playback_cfg, recording_dir, max_frames)

    if cfg.processes != 1 and playback_cfg.show_gui:
        log.warning("Multiprocess rendering requires headless mode, setting cfg.show_gui = False")
        playback_cfg.show_gui = False
    # in order to distribute the rendering to multiple processes, predetermine the lengths of the
    # (rendered) episodes and to which (recording) file ids the episode start and end correspond
    # a rendered episode does not contain the done frame, thus length(render_episode) = length(recording_episode) -1
    episode_lengths, render_start_end_ids = get_episode_lengths(cfg.load_dir, max_frames)
    # episode_lengths = episode_lengths[:1]
    # render_start_end_ids = render_start_end_ids[:1]

    if cfg.processes > len(episode_lengths):
        log.warning(f"Trying to use more processes ({cfg.processes}) than episodes ({len(episode_lengths)}).")
        log.warning(f"Reducing number of processes to {len(episode_lengths)}.")
        cfg.processes = len(episode_lengths)
    # distribute the episodes equally to processes
    split_indices = np.array_split(np.array(render_start_end_ids), cfg.processes, axis=0)
    # every process renders the interval [proc_start_ids, proc_end_ids)
    proc_start_ids = [split_indices[proc_num][0][0] for proc_num in range(cfg.processes)]
    proc_end_ids = [split_indices[proc_num][-1][1] for proc_num in range(cfg.processes)]
    # predetermine starting episode indices for multiple processes
    proc_ep_ids = np.cumsum(
        [0] + list(map(np.sum, np.array_split(np.array(episode_lengths), cfg.processes, axis=0)))[:-1]
    )
    proc_ep_ids += num_prev_rendered_episodes
    if cfg.processes > 1:
        processes = [
            Process(
                target=worker_run,
                args=(
                    recording_dir,
                    playback_cfg,
                    proc_num,
                    proc_start_ids[proc_num],
                    proc_end_ids[proc_num],
                    proc_ep_ids[proc_num],
                ),
                name=f"Worker {proc_num}",
            )
            for proc_num in range(cfg.processes)
        ]
        deque(map(lambda proc: proc.start(), processes))
        deque(map(lambda proc: proc.join(), processes))
    else:
        worker_run(recording_dir, playback_cfg, 0, 0, max_frames, num_prev_rendered_episodes)
    save_ep_lens(episode_lengths, num_prev_rendered_episodes)

    log.info("All workers done")


def build_rendering_config(recording_dir, rendering_config):
    merged_conf = omegaconf.OmegaConf.load(Path(recording_dir) / ".hydra" / "config.yaml")
    merged_conf = omegaconf.OmegaConf.merge(merged_conf, rendering_config)
    hydra.core.utils._save_config(merged_conf, "merged_config.yaml", Path(os.getcwd(), ".hydra"))
    return merged_conf


def load_rendering_config(rendering_config):
    conf = omegaconf.OmegaConf.load(Path(os.getcwd()) / ".hydra" / "merged_config.yaml")
    override_conf = omegaconf.OmegaConf.select(rendering_config, "scene")
    omegaconf.OmegaConf.update(conf, "scene", override_conf, merge=False)
    conf.set_static_cam = False
    return conf


def num_previously_rendered_episodes():
    return len(list(Path(os.getcwd()).glob("*.npz")))


def save_ep_lens(episode_lengths, num_prev_episodes):
    if num_prev_episodes > 0:
        previous_ep_lens = np.load("ep_lens.npy")
        episode_lengths = np.concatenate((previous_ep_lens, episode_lengths))
    np.save("ep_lens.npy", episode_lengths)
    end_ids = np.cumsum(episode_lengths) - 1
    start_ids = [0] + list(end_ids + 1)[:-1]
    ep_start_end_ids = list(zip(start_ids, end_ids))
    np.save("ep_start_end_ids.npy", ep_start_end_ids)


def save_step(counter, rgbs, depths, actions, robot_obs, scene_obs, cam_names, **additional_infos):
    rgb_entries = {f"rgb_{cam_name}": rgbs[f"rgb_{cam_name}"] for i, cam_name in enumerate(cam_names)}
    depths_entries = {f"depth_{cam_name}": depths[f"depth_{cam_name}"] for i, cam_name in enumerate(cam_names)}
    if actions[-1] == 0:
        actions[-1] = -1
    np.savez_compressed(
        f"episode_{counter:07d}.npz",
        actions=actions,
        rel_actions=to_relative_action(actions, robot_obs),
        robot_obs=robot_obs,
        scene_obs=scene_obs,
        **rgb_entries,
        **depths_entries,
        **additional_infos,
    )


def state_to_action(info):
    """
    save action as [tcp_pos, tcp_orn_quaternion, gripper_action]
    """
    tcp_pos = info["robot_info"]["tcp_pos"]
    tcp_orn = info["robot_info"]["tcp_orn"]
    gripper_action = info["robot_info"]["gripper_action"]
    action = np.concatenate([tcp_pos, tcp_orn, [gripper_action]])
    return action


def set_static_cams_from_gui(cfg, load_dir, max_frames):
    import cv2

    assert cfg.env.show_gui
    env = hydra.utils.instantiate(cfg.env)
    env.reset()
    frame = 0
    log.info("--------------------------------------------------")
    log.info("Use Debug GUI to change the position of the camera")
    log.info("Use Render_view_window for keyboard input")
    log.info("Press A or D to move through frames")
    log.info("Press Q or E to skip through frames")
    log.info("Press S to set camera position")
    log.info("Press ENTER to save the set camera position")
    log.info("Press ESC to skip setting position for current camera")
    for cam_index, (cam_name, cam) in enumerate(cfg.cameras.items()):
        if "static" in cam._target_:
            # initialize variables
            look_from = cam.look_from
            look_at = cam.look_at
            up_vector = cam.up_vector
            fov = cam.fov
            while True:
                file_path = load_dir / f"{frame:012d}.pickle"
                _, _, _ = env.reset_from_storage(file_path)
                env.p.stepSimulation()
                frame_rgbs, frame_depths = env.get_camera_obs()
                rgb_static = frame_rgbs[cam_index]

                cv2.imshow("Render_view_window", cv2.resize(rgb_static, (500, 500))[:, :, ::-1])
                k = cv2.waitKey(10) % 256
                if k == ord("a"):
                    frame -= 1
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("d"):
                    frame += 1
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("q"):
                    frame -= 100
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("e"):
                    frame += 100
                    frame = np.clip(frame, 0, max_frames - 1)
                if k == ord("z"):
                    c = env.cameras[cam_index]
                    c.fov -= 1
                    c.projectionMatrix = p.computeProjectionMatrixFOV(
                        fov=c.fov, aspect=c.aspect, nearVal=c.nearval, farVal=c.farval
                    )
                    print(c.fov)
                    fov = c.fov
                if k == ord("x"):
                    c = env.cameras[cam_index]
                    c.fov += 1
                    c.projectionMatrix = p.computeProjectionMatrixFOV(
                        fov=c.fov, aspect=c.aspect, nearVal=c.nearval, farVal=c.farval
                    )
                    print(c.fov)
                    fov = c.fov
                if k == ord("r"):
                    c = env.cameras[cam_index]
                    direction_vector = np.array(c.look_at) - np.array(c.look_from)
                    c.up_vector = (
                        R.from_rotvec(0.1 * direction_vector / np.linalg.norm(direction_vector)).as_matrix()
                        @ c.up_vector
                    )
                    up_vector = c.up_vector
                if k == ord("f"):
                    c = env.cameras[cam_index]
                    direction_vector = np.array(c.look_at) - np.array(c.look_from)
                    c.up_vector = (
                        R.from_rotvec(-0.1 * direction_vector / np.linalg.norm(direction_vector)).as_matrix()
                        @ c.up_vector
                    )
                    up_vector = c.up_vector
                if k == 13:  # Enter
                    cam.look_from = look_from
                    cam.look_at = look_at
                    log.info(f"Set look_from of camera {cam_index} to {look_from}")
                    log.info(f"Set look_at of camera {cam_index} to {look_at}")
                    cam.up_vector = np.array(up_vector).tolist()
                    log.info(f"Set up_vector of camera {cam_index} to {up_vector}")
                    cam.fov = fov
                    log.info(f"Set fov of camera {cam_index} to {fov}")
                    break
                if k == 27:  # ESC
                    log.info(f"Do no change position of camera {cam_index}")
                    break
                # if k == ord("s"):
                look_from, look_at = env.cameras[cam_index].set_position_from_gui()
    hydra.core.utils._save_config(cfg, "merged_config.yaml", Path(os.getcwd(), ".hydra"))
    env.close()
    return cfg


def worker_run(load_dir, rendering_cfg, proc_num, start_frame, stop_frame, episode_index):
    log.info(f"[{proc_num}] Starting worker {proc_num}")
    set_egl_device(0)
    env = hydra.utils.instantiate(rendering_cfg.env)

    log.info(f"[{proc_num}] Entering Loop")
    frame_counter = 0
    rgbs, depths, actions, robot_obs, scene_obs, = (
        [],
        [],
        [],
        [],
        [],
    )
    for frame in range(start_frame, stop_frame):
        file_path = load_dir / f"{frame:012d}.pickle"
        state_ob, done, info = env.reset_from_storage(file_path)
        action = state_to_action(info)
        robot_obs.append(state_ob["robot_obs"])
        scene_obs.append(state_ob["scene_obs"])

        # action is robot state of next frame
        if frame_counter > 0:
            actions.append(action)
        frame_rgbs, frame_depths = env.get_camera_obs()
        rgbs.append(frame_rgbs)
        depths.append(frame_depths)
        # for terminal states save current robot state as action
        frame_counter += 1
        log.debug(f"episode counter {episode_index} frame counter {frame_counter} done {done}")

        if frame_counter > 1:
            save_step(
                episode_index,
                rgbs.pop(0),
                depths.pop(0),
                actions.pop(0),
                robot_obs.pop(0),
                scene_obs.pop(0),
                cam_names=[cam.name for cam in env.cameras],
            )
            episode_index += 1
            if done:
                frame_counter = 0
                rgbs, depths, actions, robot_obs, scene_obs = [], [], [], [], []

        log.debug(f"[{proc_num}] Rendered frame {frame}")

    assert done

    env.close()
    log.info(f"[{proc_num}] Finishing worker {proc_num}")


if __name__ == "__main__":
    main()
