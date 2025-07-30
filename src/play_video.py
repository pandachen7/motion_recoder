import cv2


def play_video(video_path):
    """
    使用 OpenCV 播放视频。

    Args:
        video_path: 视频文件的路径。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("视频播放结束或文件损坏。")
            break

        cv2.imshow("Video Player", frame)

        # 按 'q' 键退出播放
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\marcy\Downloads\drive-download-20250714T120222Z-1-001\1140328\C203_230304_003917_04_0.mp4"
    play_video(video_path)
