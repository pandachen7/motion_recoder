from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()


class Config:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} 不允許被實例化，請直接使用類別屬性")

    with Path("./cfg/system.yaml").open() as file:
        __yaml_sys = yaml.load(file)

    video_source: str = __yaml_sys["source"]
    record_source: str = __yaml_sys.get("record_source", None)

    sample_per_n_second: int = __yaml_sys["sample_per_n_second"]
    max_buff_frames: int = __yaml_sys["max_buff_frames"]
    mask_path: str = __yaml_sys.get("mask_path")

    motion_area_threshold: int = __yaml_sys.get("motion_area_threshold", 3000)

    use_external_hd: bool = __yaml_sys.get("use_external_hd", False)
    select_partition_device: str = __yaml_sys.get("select_partition_device", "")
    path_out_external: str = __yaml_sys.get("path_out_external", "")
    path_out_local: str = __yaml_sys.get("path_out_local", "./recordings")

    record_seconds: int = __yaml_sys.get("record_seconds", 30)

    rec_method: str = __yaml_sys.get("rec_method", "opencv")

    rec_audio: bool = __yaml_sys.get("rec_audio", False)
    copy_stream: bool = __yaml_sys.get("copy_stream", True)
    ffmpeg_fps: int = __yaml_sys.get("ffmpeg_fps", 5)

    fourcc: str = __yaml_sys.get("fourcc", "XVID")
    opencv_fps: int = __yaml_sys.get("opencv_fps", 30)

    imshow_enabled: bool = __yaml_sys.get("imshow", False)
    win_width, win_height = __yaml_sys.get("window_size", [640, 480])

    schedule_config: dict = __yaml_sys.get("schedule", {})

    if not isinstance(sample_per_n_second, int) or sample_per_n_second < 1:
        sample_per_n_second = 1
