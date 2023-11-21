import logging
import multiprocessing as mp
import os
import pickle
import time

# A logger for this file
log = logging.getLogger(__name__)


class DataRecorder:
    """
    Collects frame information to file in output-dir with
    filename: <TIMESTAMP>/<FRAME>.pickle
    Saving facility with separate worker thread.
    """

    def __init__(self, env, record_fps, enable_tts):
        """
        Setup MultiprocessingStorage
        """
        self.env = env
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = 0
        log.info("Starting serialization worker process")
        self.prev_time = time.time()
        self.loop_time = 1.0 / record_fps
        self._unsaved_vr_events = []
        self.enable_tts = enable_tts
        if enable_tts:
            import pyttsx3

            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 175)
        self.prev_done = False
        self.current_episode_filenames = []

    def step(self, prev_vr_event, state_obs, done, info):
        self._unsaved_vr_events.extend(prev_vr_event)
        current_time = time.time()
        delta_t = current_time - self.prev_time
        if delta_t >= self.loop_time or done:
            log.debug(f"Record FPS: {1 / delta_t:.0f}")
            self.prev_time = time.time()
            file_path = f"{str(self.save_frame_cnt).zfill(12)}.pickle"
            self.save(file_path, self._unsaved_vr_events, state_obs, done, info)
            if self.prev_done and not done:
                self.current_episode_filenames = []
            self.current_episode_filenames.append(file_path)

            # file_path_state = f"{str(self.save_frame_cnt).zfill(12)}_state.pickle"
            # with open(file_path_state, 'wb') as file:
            #     pickle.dump(state_obs, file)

            self._unsaved_vr_events = []
            self.save_frame_cnt += 1
            self.prev_done = done

    def save(self, filename, vr_events, state_obs, done, info):
        """
        Extract dataFrame from pybullet and enqueue for worker thread.

        Args:
            filename: path to file
            vr_events: vrEvents to attach to data
            state_obs: state observations
            done: true if episode ends
            info: info dict

        Returns:
            None
        """
        data = self.env.serialize()
        data["vr_events"] = vr_events
        data["state_obs"] = state_obs
        data["done"] = done
        data["info"] = info
        self.queue.put((filename, data))

    def delete_episode(self):
        num_frames = len(self.current_episode_filenames)
        if self.enable_tts:
            self.tts.say(f"Deleting last episode with {num_frames} frames")
            self.tts.runAndWait()
        for filename in self.current_episode_filenames:
            os.remove(filename)
        if self.enable_tts:
            self.tts.say("Finished deleting")
            self.tts.runAndWait()
        self.save_frame_cnt -= num_frames
        self.current_episode_filenames = []

    def process_queue(self):
        """
        Process function for queue.
        Returns:
            None
        """
        while True:
            msg = self.queue.get()
            if msg == "QUIT":
                self.running = False
                break
            (filename, data) = msg
            with open(filename, "wb") as file:
                pickle.dump(data, file)

    def close(self):
        """
        Tell Worker to shut down.
        Returns:
            None
        """
        if self.running:
            self.queue.put("QUIT")
            self.process.join()

    def __enter__(self):
        """
            with ... as ... : logic
        Returns:
            None
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            with ... as ... : logic
        Returns:
            None
        """
        self.close()
