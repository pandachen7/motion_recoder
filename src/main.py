import os
import threading
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import cv2
import numpy as np
import psutil

from config import Config
from loglo import getUniqueLogger

log = getUniqueLogger(__file__)


FORMAT_DATE = "%Y-%m-%d"
FORMAT_TIME = "%y%m%d_%H%M%S_%f"


def getDate():
    # get 2025-06-12 which means 2025/06/12
    return datetime.now().strftime(FORMAT_DATE)


def writeFont(img, text, x, y, color=(0, 255, 255), font_scale=1, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    shadow = (0, 0, 0)

    # 左上角的座標 (注意y是baseline位置)
    x, y = 10, 30  # 讓字不會貼在邊界
    cv2.putText(
        img, text, (x + 2, y + 2), font, font_scale, shadow, thickness, cv2.LINE_AA
    )
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


class MotionDetector:
    def __init__(self):
        self.loaded = False
        self.proc = None

        self.output_path = "./recordings"

        # rtmp, http沒試過
        if Config.record_source:
            self.ffmpeg_source = Config.record_source
        else:
            self.ffmpeg_source = Config.video_source
        if Config.video_source.startswith("/dev/video"):
            self.video_type = "stream"
        elif "://" in Config.video_source:
            self.video_type = "stream"
        else:
            self.video_type = "file"

        self.cap = cv2.VideoCapture(Config.video_source)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video source.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = Config.opencv_fps

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.d(f"fps: {self.fps}, width: {self.width}, height: {self.height}")

        self.mask = None
        if Config.mask_path and os.path.exists(Config.mask_path):
            # mask = cv2.imread(Config.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(Config.mask_path, cv2.IMREAD_UNCHANGED)
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

            self.mask = mask
            if Config.imshow_enabled:
                resized = cv2.resize(
                    mask,
                    (Config.win_width, Config.win_height),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("mask", resized)
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
            folder_root = Config.path_out_local
            if Config.use_external_hd:
                if Config.path_out_external and Path(Config.path_out_external).is_dir():
                    folder_root = Config.path_out_external
                elif Config.select_partition_device:
                    partitions = psutil.disk_partitions()
                    for p in partitions:
                        if (
                            p.device == Config.select_partition_device
                            and Path(p.mountpoint).is_dir()
                        ):
                            folder_root = p.mountpoint
                            break
                else:
                    log.warning(
                        f"no external hard disk detected. use local space `{Config.path_out_local}`"
                    )
                    folder_root = Config.path_out_local

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
        if not Config.schedule_config.get("enabled", False):
            return True

        now = datetime.now().time()
        start_time_conf = Config.schedule_config.get(
            "start_time", {"hour": 0, "minute": 0}
        )
        end_time_conf = Config.schedule_config.get(
            "end_time", {"hour": 23, "minute": 59}
        )

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

            t1 = time.time()
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
            if current_time - last_sample_time >= Config.sample_per_n_second:
                last_sample_time = current_time
                self.frame_buffer.append(gray)
                if len(self.frame_buffer) > Config.max_buff_frames:
                    self.frame_buffer.pop(0)

            if len(self.frame_buffer) > 0:
                self.background = np.median(np.array(self.frame_buffer), axis=0).astype(
                    np.uint8
                )

            motion_area = 0
            if self.background is not None:
                frame_delta = cv2.absdiff(self.background, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

                if self.mask is not None:
                    thresh = cv2.bitwise_and(thresh, thresh, mask=self.mask)

                contours, _ = cv2.findContours(
                    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    motion_area = max(motion_area, cv2.contourArea(contour))
                # log.d(motion_area)

                if motion_area > Config.motion_area_threshold:
                    self.last_motion_time = time.time()
                    if not self.is_recording:
                        self.start_recording()

                if self.is_recording:
                    if self.video_writer is not None:
                        self.video_writer.write(frame)
                    if time.time() - self.last_motion_time > Config.record_seconds:
                        self.stop_recording()

                if Config.imshow_enabled:
                    resized = cv2.resize(
                        frame,
                        (Config.win_width, Config.win_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow("frame", resized)

                    resized = cv2.resize(
                        thresh,
                        (Config.win_width, Config.win_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    writeFont(
                        resized, f"max_area: {motion_area}", 10, 20, color=(255, 255, 255)
                    )
                    cv2.imshow("Thresh", resized)

                    if self.background is not None:
                        resized = cv2.resize(
                            self.background,
                            (Config.win_width, Config.win_height),
                            interpolation=cv2.INTER_AREA,
                        )
                        cv2.imshow("Background", resized)
            log.d(
                f"motion detect fps: {1 / (time.time() - t1):.2f}, max motion area: {motion_area:.0f}"
            )
            if Config.imshow_enabled and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop_recording()
        self.cap.release()
        if Config.imshow_enabled:
            cv2.destroyAllWindows()

    def start_recording(self):
        self.is_recording = True
        self._updateFolder()
        output_dir = self.output_path

        now = datetime.now()
        time_str = now.strftime("%H_%M_%S")
        filename = os.path.join(output_dir, f"{time_str}.mp4")

        if Config.rec_method == "ffmpeg":
            cmd = ["ffmpeg"]
            cmd += [
                "-i",
                self.ffmpeg_source,
                "-t",
                str(Config.record_seconds),
                # "-preset",
                # "veryfast",
            ]
            if not Config.rec_audio:
                cmd += ["-an"]
            if Config.copy_stream:
                cmd += ["-c", "copy"]
            else:
                cmd += [
                    # "-c:v",
                    # "libx264",
                    # "-crf",
                    # "18",
                    "-r",
                    str(Config.ffmpeg_fps),
                ]
            cmd += [filename]

            log.d(f"ffmpeg cmd: {cmd}")
            self.proc = Popen(cmd)
        elif Config.rec_method == "opencv":
            fourcc = cv2.VideoWriter_fourcc(*Config.fourcc)
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, self.fps, (self.width, self.height)
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
