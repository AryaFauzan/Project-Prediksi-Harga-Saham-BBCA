@echo off
echo Memulai proses training model BBCA...
cd /d "%~dp0"
python train_model.py
echo Training selesai!
pause
