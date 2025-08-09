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
