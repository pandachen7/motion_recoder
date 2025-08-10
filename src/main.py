import os
import threading
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import cv2
import numpy as np
import psutil
from ruamel.yaml import YAML

from loglo import getUniqueLogger

log = getUniqueLogger(__file__)
yaml = YAML()

FORMAT_DATE = "%Y-%m-%d"
FORMAT_TIME = "%y%m%d_%H%M%S_%f"


def getDate():
    # get 2025-06-12 which means 2025/06/12
    return datetime.now().strftime(FORMAT_DATE)


class MotionDetector:
    def __init__(self, config_path="./cfg/config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.load(f)

        self.loaded = False
        self.proc = None

        self.video_source = self.config["source"]
        self.video_fps = self.config.get("video_fps", 30)
        self.sample_per_second = self.config["sample_per_second"]
        self.max_buff_frames = self.config["max_buff_frames"]
        self.mask_path = self.config.get("mask_path")

        self.motion_area_threshold = self.config["motion_area_threshold"]

        self.output_path = "./recordings"
        self.use_external_hd = self.config.get("use_external_hd", False)
        self.select_partition_device = self.config.get("select_partition_device", "")
        self.path_out_external = self.config.get("path_out_external", "")
        self.path_out_local = self.config.get("path_out_local", "./recordings")

        self.rec_method = self.config.get("rec_method", "opencv")
        self.rec_audio = self.config.get("rec_audio", False)
        # self.use_hwaccel = self.config.get("use_hwaccel", None)
        self.record_fps = self.config.get("record_fps", 5)
        self.record_seconds = self.config.get("record_seconds", 30)

        self.fourcc = cv2.VideoWriter_fourcc(*self.config["fourcc"])

        self.imshow_enabled = self.config.get("imshow", False)

        self.schedule_config = self.config.get("schedule", {})

        if self.sample_per_second < 1:
            self.sample_per_second = 1

        # rtmp, http沒試過
        self.ffmpeg_source = self.video_source
        if self.video_source.startswith("/dev/video"):
            self.video_type = "stream"
        elif "://" in self.video_source:
            self.video_type = "stream"
        else:
            self.video_type = "file"

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
            # mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (self.width, self.height))
            if len(mask.shape) == 2:
                # 灰階圖（單通道）, 0 是 mask，其餘為非 mask, 未測試
                mask = np.where(mask == 0, 0, 255).astype(np.uint8)
            elif mask.shape[2] == 3:
                pass
                # 彩色圖（3通道）
                black = np.all(mask == [0, 0, 0], axis=2)
                mask = np.where(black, 0, 255).astype(np.uint8)
            elif mask.shape[2] == 4:
                # 含 alpha 的圖（4通道）
                rgb = mask[:, :, :3]
                alpha = mask[:, :, 3]
                black_and_opaque = np.all(rgb == [0, 0, 0], axis=2) & (alpha == 255)
                fully_transparent = alpha == 0
                mask = np.where(black_and_opaque, 0, 255)
                mask[fully_transparent] = 255
                mask = mask.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {mask.shape}")

            resized_mask = mask
            cv2.resize(resized_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Mask Sample", resized_mask)

            cv2.waitKey(0)
            self.mask = mask
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

        self._day_now = datetime.now().day - 1

        self._updateFolder()

    def _updateFolder(self):
        # this is for unified folder
        if self._day_now != datetime.now().day:
            self._day_now = datetime.now().day
            folder_root = self.path_out_local
            if self.use_external_hd:
                if self.select_partition_device:
                    partitions = psutil.disk_partitions()
                    for p in partitions:
                        if (
                            p.device == self.select_partition_device
                            and Path(p.mountpoint).is_dir()
                        ):
                            folder_root = p.mountpoint
                            break
                elif self.path_out_external and Path(self.path_out_external).is_dir():
                    folder_root = self.path_out_external
                else:
                    log.warning(
                        f"no external hard disk detected. use local space `{self.path_out_local}`"
                    )
                    folder_root = self.path_out_local

            self.output_path = Path(folder_root, getDate()).as_posix()
            log.info(f"change folder to {self.output_path}")

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

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

    def _is_in_schedule(self):
        if not self.schedule_config.get("enabled", False):
            return True

        now = datetime.now().time()
        start_time_conf = self.schedule_config.get(
            "start_time", {"hour": 0, "minute": 0}
        )
        end_time_conf = self.schedule_config.get("end_time", {"hour": 23, "minute": 59})

        start_time = now.replace(
            hour=start_time_conf["hour"],
            minute=start_time_conf["minute"],
            second=0,
            microsecond=0,
        )
        end_time = now.replace(
            hour=end_time_conf["hour"],
            minute=end_time_conf["minute"],
            second=59,
            microsecond=999999,
        )

        return start_time <= now <= end_time

    def run(self):
        last_sample_time = time.time()

        while True:
            if not self._is_in_schedule():
                if self.is_recording:
                    log.i("duty off")
                    self.stop_recording()
                time.sleep(60)  # 不在排程內
                continue

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
                    motion_area = max(motion_area, cv2.contourArea(contour))
                # log.d(motion_area)

                if motion_area > self.motion_area_threshold:
                    self.last_motion_time = time.time()
                    if not self.is_recording:
                        self.start_recording()

                if self.is_recording:
                    if self.video_writer is not None:
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
        self._updateFolder()
        output_dir = self.output_path

        now = datetime.now()
        time_str = now.strftime("%H_%M_%S")
        filename = os.path.join(output_dir, f"{time_str}.mp4")

        if self.rec_method == "ffmpeg":
            cmd = ["ffmpeg"]
            # if self.use_hwaccel == "qsv":
            #     # cmd += ["-hwaccel", "qsv", "-c:v", "h264_qsv"]
            #     cmd += ["-hwaccel", "qsv"]
            cmd += [
                "-i",
                self.ffmpeg_source,
                "-t",
                str(self.record_seconds),
                "-preset",
                "veryfast",
                # "-r",
                # str(self.record_fps),
            ]
            if not self.rec_audio:
                cmd += ["-an"]
            cmd += ["-c:v", "copy", filename]  # 直接copy到檔案

            log.d(f"ffmpeg cmd: {cmd}")
            self.proc = Popen(cmd)
        elif self.rec_method == "opencv":
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
            if self.proc is not None:
                log.w("ffmpeg process still running")
                # proc.stdin.write(b'q')
                # proc.stdin.flush()
                # <OR>
                # 發送 SIGTERM 信號給 FFmpeg 進程
                # os.kill(proc.pid, signal.SIGTERM)
                # <OR>
                # self.proc.terminate()
                # self.proc.wait()
                # self.proc = None
            log.i("Stopped recording.")


if __name__ == "__main__":
    detector = MotionDetector()
    detector.run()
