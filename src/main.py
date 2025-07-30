import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ruamel.yaml import YAML

from loglo import getUniqueLogger

log = getUniqueLogger(__file__)
yaml = YAML()


class MotionDetector:
    def __init__(self, config_path="./cfg/config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.load(f)

        self.loaded = False

        self.video_source = self.config["source"]
        self.video_fps = self.config.get("video_fps", 20)
        self.motion_area_threshold = self.config["motion_area_threshold"]
        self.output_path = self.config["output_path"]
        self.sample_per_second = self.config["sample_per_second"]
        self.max_buff_frames = self.config["max_buff_frames"]
        self.mask_path = self.config.get("mask_path")
        self.imshow_enabled = self.config["imshow"]
        self.record_seconds = self.config.get("record_seconds", 5)
        self.fourcc = cv2.VideoWriter_fourcc(*self.config["fourcc"])

        if self.sample_per_second < 1:
            self.sample_per_second = 1
        self.video_type = "stream" if "://" in self.video_source else "file"

        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path, exist_ok=True)
            except OSError:
                log.d("Could not create output directory. use `./recordings` instead.")
                self.output_path = "./recordings"
                os.makedirs(self.output_path, exist_ok=True)

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video source.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = self.video_fps

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.d(f"fps: {self.fps}, width: {self.width}, height: {self.height}")

        self.mask = None
        if self.mask_path and os.path.exists(self.mask_path):
            self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            self.mask = cv2.resize(self.mask, (self.width, self.height))
            log.d("load mask successfully")

        self.frame_buffer = []
        self.background = None
        self.is_recording = False
        self.video_writer = None
        self.last_motion_time = 0

        self.latest_frame = None
        self.lock = threading.Lock()
        if self.video_type == "stream":
            # self.latest_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.thread = threading.Thread(target=self._frame_reader, daemon=True)
            self.thread.start()

    def _frame_reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.latest_frame = frame

    def toNextFrame(self):
        if self.video_type == "file":
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                self.latest_frame = None

    def run(self):
        last_sample_time = time.time()

        while True:
            self.toNextFrame()
            with self.lock:
                if self.latest_frame is None:
                    if self.video_type == "stream":
                        continue
                    else:
                        break
                frame = self.latest_frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            current_time = time.time()
            if current_time - last_sample_time >= 1.0 / self.sample_per_second:
                last_sample_time = current_time
                self.frame_buffer.append(gray)
                if len(self.frame_buffer) > self.max_buff_frames:
                    self.frame_buffer.pop(0)

            if len(self.frame_buffer) > 0:
                self.background = np.median(np.array(self.frame_buffer), axis=0).astype(
                    np.uint8
                )

            if self.background is not None:
                frame_delta = cv2.absdiff(self.background, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

                if self.mask is not None:
                    thresh = cv2.bitwise_and(thresh, thresh, mask=self.mask)

                contours, _ = cv2.findContours(
                    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                motion_area = 0
                for contour in contours:
                    motion_area += cv2.contourArea(contour)

                if motion_area > self.motion_area_threshold:
                    self.last_motion_time = time.time()
                    if not self.is_recording:
                        self.start_recording()

                if self.is_recording:
                    self.video_writer.write(frame)
                    if time.time() - self.last_motion_time > self.record_seconds:
                        self.stop_recording()

                if self.imshow_enabled:
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Thresh", thresh)
                    if self.background is not None:
                        cv2.imshow("Background", self.background)

            if self.imshow_enabled and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop_recording()
        self.cap.release()
        if self.imshow_enabled:
            cv2.destroyAllWindows()

    def start_recording(self):
        self.is_recording = True
        now = datetime.now()
        date_path = now.strftime("%Y_%m_%d")
        output_dir = os.path.join(self.output_path, date_path)
        os.makedirs(output_dir, exist_ok=True)

        time_str = now.strftime("%H_%M_%S")
        filename = os.path.join(output_dir, f"{time_str}.avi")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(
            filename, self.fourcc, self.fps, (self.width, self.height)
        )
        log.i(f"Started recording: {filename}")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                log.i("Stopped recording.")


if __name__ == "__main__":
    try:
        detector = MotionDetector()
        detector.run()
    except Exception as e:
        print(f"An error occurred: {e}")
