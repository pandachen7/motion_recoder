mask可讀取三種圖, 單通道, RGB, RGBA
- 單通道: 全黑為遮蔽 (即pixel亮度為0才會遮蔽)
- RGB: 全黑為遮蔽, 其他通過. 因此<0, 0, 1> 也是通過
- RGBA: 當RGB都是0, alpha為255時, 才為遮蔽; 其他全通過

注意mask永遠都是單通道

# 以下開機自動執行與掛載行動硬碟
開創要mount的資料夾
`sudo mkdir /media/ex_hdd`
一般設定fstab會設定硬碟UUID, 但如果不綁定特定硬碟:
`sudo nano /etc/fstab`
```bash
/dev/sda2            /media/ex_hdd         ext4           defaults
# <OR>
/dev/sda1             /media/ex_hdd         vfat           nofail,noauto,x-systemd.automount,uid=1000,gid=1000,dmask=000,fmask=111,user    0 2
# <OR>
UUID=$(blkid -s UUID -o value /dev/sda1)       /media/ex_hdd         vfat           nofail,noauto,x-systemd.automount,uid=1000,gid=1000,dmask=000,fmask=111,user    0 2
# 或用lsblk -f 來查詢

# 空白自動加減對齊
```
用完fstab之後用sudo mount -a試看看, 再用 df -h 應該能看到掛載的

## systemd
以下簡短範例
```
[Unit]
Description=Auto Webcam/RTSP Recording
After=remote-fs.target

[Service]
ExecStart=/usr/local/bin/record.sh
Restart=always
User=你的使用者
```

## ffmpeg
```
ffmpeg -hwaccel qsv -c:v h264_qsv -i input.mp4 -c:v h264_qsv -preset fast -crf 23 -an output.mp4
# -hwaccel: 電腦加速解碼 x555ln之類可能需要
# -c:v 解碼, 非必要
# -i: 可以直接改rtsp路徑, 或usb cam的設備名稱
# -c:v 編碼, 只處理video部分, 如果影音都要就 -c 即可
# -preset: fast, veryfast
# -crf 23: 數值愈小品質愈好
# -an: 禁音

# 直接pass, 但無法改變fps與品質
ffmpeg -f v4l2 -i XXX -c:v copy out.mp4

# 使用x264來錄影30秒, 5 fps
ffmpeg -i /dev/video0 -t 30 -r 5 -c:v libx264 -preset fast output.mp4
```

### webcam共用
```
sudo modprobe v4l2loopback video_nr=30 exclusive_caps=1 card_label="loopback_cam"
ffmpeg -f v4l2 -i /dev/video0 -codec copy -f v4l2 /dev/video30
```

### 查詢rtsp編碼格式, 一般來說ffmpeg就已經會自動偵測編碼格式
```
ffprobe -v quiet -show_streams rtsp://your_rtsp_url
```