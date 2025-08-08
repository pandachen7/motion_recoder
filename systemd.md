# systemd

## 一些指令

凡修改過就要載入
`sudo systemctl daemon-reload`

若要紀錄log, 記得先開log的路徑, e.g.

`sudo mkdir /var/log/uvicorn`

### 設定新的system service

`sudo systemctl edit --force --full uvicorn`

以後就能用

`sudo systemctl edit --full uvicorn`

來編輯, 簡單模板如下, 基本上大多數的環境變數可以加在.env中, 再由程式自己讀

```bash
[Unit]
Description=uvicorn
After=network.target

[Service]
User=asys
Type=simple
Environment="PATH=/home/asys/.local/bin:/usr/bin"
Environment=PYTHONUNBUFFERED=1

WorkingDirectory=/home/asys/ws/llm
PrivateTmp=true

# ExecStartPre=/opt/your_command
ExecStart=/home/asys/ws/venv/llm_rag/bin/python src/main
# ExecStartPost=/opt/your_command

ExecReload=  
RestartSec=2s
Restart=always

# 改用預設的journal
StandardOutput=append:/var/log/uvicorn/log
StandardError=append:/var/log/uvicorn/loge

[Install]
WantedBy=multi-user.target
```

設定測試 (後面的.service會自己加上)

`SERVICE_NAME=uvicorn`

`SERVICE_NAME=express_umijs`

`sudo systemctl start $SERVICE_NAME`

`sudo systemctl stop $SERVICE_NAME`

`sudo systemctl status $SERVICE_NAME`
如果實在無法在任何地方找到錯誤跡象, 試試daemon-reload -> stop -> start

開機自動執行
`sudo systemctl enable $SERVICE_NAME`

`sudo systemctl disable $SERVICE_NAME`

關於設定, 假設venv環境在 `/home/asys/ws/venv/llm_api/bin/`

特殊環境, 在python venv執行.sh時, 需要用

```bash
ExecStart=/bin/bash -c 'source /home/asys/ws/venv/stable_diffuser/bin/activate && /home/asys/ws/sd_webui/webui.sh --listen'
# OR
ExecStart=/bin/bash -c '/home/asys/ws/venv/XXX/bin/python /home/asys/ws/sd_webui/XXX.py --listen'
```

直接source然後&&跑.sh, 這大概是最簡便的方式了
因為設定PYTHONPATH似乎沒用

## system的log預設紀錄到Journal

為防止檔案變得太大, 先設定 `sudo nano /etc/systemd/journald.conf`

```bash
[Journal]
Storage=persistent
SystemMaxUse=500M
MaxRetentionSec=6month
```

### journalctl導出log

如果想每個月都導出一個log檔, 就先建立一個cron job, 記得先

`sudo mkdir /var/log/journal-export`

然後

`sudo nano /usr/local/bin/export_journal_monthly.sh`

```bash
#!/bin/bash
last_month=$(date --date="$(date +%Y-%m-15) -1 month" +%Y-%m)
start="${last_month}-01"
end=$(date -d "$start +1 month" +%Y-%m-%d)
journalctl --since "$start" --until "$end" > /var/log/journal-export/systemd-${last_month}.log
```

設定 crontab, 使用自己的帳號即可, 不用sudo

`crontab -e`

`0 1 1 * * /usr/local/bin/export_journal_monthly.sh`

系統會自動偵測並且排程

### journalctl指令

```bash
# 查看單一服務log
journalctl -u $SERVICE_NAME
# 查看一個範圍
journalctl --since "2025-07-01" --until "2025-07-15"
journalctl -b -1               # 查看前一次開機的 log
# journald雖然有搜尋, 過濾等優勢, 但無法上色, 想要觀看顏色就要用複雜指令e.g.
journalctl -u myapp --no-pager --output=cat | less -R
# 如果要用簡寫, 那可以在.bashrc加上, 就能
jc() {
  journalctl -u "$@" --no-pager --output=cat | less -R
}
jce() {
  journalctl -u "$@" _TRANSPORT=stderr --no-pager --output=cat | less -R
}
```

詳細解說

```bash
[Unit]
Description=uvicorn
After=network.target
# 也能加上有IP才運作的, 但沒有IP就不能運作
# After=network-online.target
# Wants=network-online.target

[Service]
User=asys
# 建議加在.env裡即可
Environment="KEY_NAME=VALUE_WITHOUT_QUOTE"
# stdout/stderr直接輸出而不buffered
Environment=PYTHONUNBUFFERED=1
Type=simple

## 如果是個人資料夾則不建議使用
# DynamicUser=true

WorkingDirectory=/home/asys/ws/llm
# 增加/tmp或/var/tmp安全性
PrivateTmp=true

## 啟動服務前，執行的指令
# ExecStartPre=/opt/your_command

ExecStart=/home/asys/ws/venv/llm_rag/bin/python src/main

## 啟動服務後，執行的指令
# ExecStartPost=/opt/your_command

## systemctl reload 有關的指令行為, `/bin/kill -HUP ${MAINPID}`為傳送中止, 但因為非同步因此不建議
ExecReload=  
RestartSec=2s
Restart=always

# 純粹想要以文字檔存檔才需要, 不然建議用預設的journald
# 服務輸出訊息導向設定 (remember use `sudo mkdir /var/log/uvicorn` first)
# 前綴append時, 才能確保重開能繼續寫入
StandardOutput=append:/var/log/uvicorn/log
StandardError=append:/var/log/uvicorn/loge

[Install]
WantedBy=multi-user.target
```