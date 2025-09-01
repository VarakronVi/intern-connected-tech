@echo off
REM ตั้งค่า conda environment และรัน main.py

REM เช็คว่า Conda อยู่ใน PATH หรือไม่
CALL conda activate yolo_stream

REM ไปยังไดเรกทอรีที่มี main.py
cd /d "."

REM รัน Python script
python main.py

REM ป้องกันหน้าต่างปิดทันที
pause