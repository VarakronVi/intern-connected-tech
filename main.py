from src.app import App
from src.backend.car_speed import SpeedEstimator, get_video_info
import cv2
from utils import *

if __name__ == "__main__":
    # ข้อมูลที่ให้มา
    config_model = {
        "Forklift": ("src/assets/models/ObjectDetection_ForkliftDetection_v1_2025_02_24.pt", 0),
        "Car": ("src/assets/models/yolov10n.pt", 2)
    }
    app = App(config_models = config_model)
    app.mainloop()