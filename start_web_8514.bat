@echo off
cd /d "d:\fyp ca1"

REM Keep server alive in a dedicated window
start "TSB-Web-8514" cmd /k "D:\anaconda\envs\fypjiu\python.exe src\web_app_tornado.py --host 0.0.0.0 --port 8514"

echo Web server started on port 8514.
echo Local:  http://127.0.0.1:8514
echo Mobile: http://172.29.0.1:8514
