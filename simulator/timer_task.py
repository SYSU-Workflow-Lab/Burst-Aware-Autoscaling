# 定时执行任务
import schedule
import time
import subprocess
def job():
    command = ["python","main.py"]
    p = subprocess.Popen(command)  # 执行命令
    p.wait()  # 等待结束

schedule.every().day.at("05:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute