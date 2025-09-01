import argparse
from collections import defaultdict, deque
from collections import namedtuple
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import threading

from src.ui.system_design import COLOR_STYLE_BGR
from utils import *

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Initialize view transformer with perspective transformation matrix."""
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an array of points using the perspective transformation."""
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class SpeedEstimator:
    def __init__(self,
                 video_info: dict,
                 confidence_threshold: float = 0.3, 
                 iou_threshold: float = 0.7,
                 yolo_model: str = "yolo11n.pt",
                 yolo_class: int = 0,
                 polygon_points: list = [[745, 209], [963, 209], [1528, 1079], [304, 1079]],
                 target_width: float = 4.6,
                 target_height: float = 23.1
                 ):       
        """Initialize SpeedEstimator with video paths and model parameters."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Define the polygon region
        self.SOURCE = np.array(polygon_points)
        self.TARGET = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ]
        )

        # Load video metadata
        self.video_info = video_info

        # Load YOLO model
        self.model = YOLO(yolo_model)
        self.yolo_class = yolo_class
        pr.yellow(f"Loaded YOLO model {yolo_model} for class {self.yolo_class}")

        # Initialize ByteTrack
        self.byte_track = sv.ByteTrack(
            frame_rate=self.video_info.fps, 
            track_activation_threshold=self.confidence_threshold
        )

        # Calculate optimal annotation parameters
        self.thickness = sv.calculate_optimal_line_thickness(resolution_wh=self.video_info.resolution_wh)
        self.text_scale = sv.calculate_optimal_text_scale(resolution_wh=self.video_info.resolution_wh)
        self.image_area = self.video_info.resolution_wh[0] * self.video_info.resolution_wh[1]
        
        self.box_annotator = sv.BoxAnnotator(thickness=self.thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.text_scale,
            text_thickness=self.thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.thickness,
            trace_length=self.video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )

        # Create PolygonZone and ViewTransformer
        self.polygon_zone = sv.PolygonZone(polygon=self.SOURCE)
        self.view_transformer = ViewTransformer(source=self.SOURCE, target=self.TARGET)

        # Dictionary to store recent positions for each tracker
        self.coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))
        
        # thread = threading.Thread(target=self.process_frame, daemon=True)
        # thread.start()
    
    def show_frame(self, frame):
        """ แสดงผลภาพที่รับเข้ามา """
        cv2.imshow("Get Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # กด 'q' เพื่อออก
            cv2.destroyAllWindows()
            
    def process_frame(self,
                      frame: np.ndarray):
        """Process video, detect objects, track movement, and estimate speed. Returns frames."""
        # Predict with YOLO
        result = self.model(frame,
                            imgsz=640,
                            classes=self.yolo_class, 
                            verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Draw polygon
        cv2.polylines(
            frame,
            [self.SOURCE.astype(np.int32)],
            isClosed=True,
            color=COLOR_STYLE_BGR.border_brand_color,
            thickness=self.thickness
        )
        
        # Filter detections
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[self.polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=self.iou_threshold)

        # Update ByteTrack trackers
        detections = self.byte_track.update_with_detections(detections=detections)

        if len(detections) > 0:
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_points = self.view_transformer.transform_points(points=points).astype(float)

            # Store recent positions for speed calculation
            for tracker_id, (x, y) in zip(detections.tracker_id, transformed_points):
                self.coordinates[tracker_id].append((x, y))
            
            # สร้างตัวกรอง `detections_filter` object ขนาดใหญ่กว่า 1 ใน 3 ของภาพให้ตัดออก
            bbox_areas = [(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in detections.xyxy]
            valid_indices = [i for i, area in enumerate(bbox_areas) if area <= (self.image_area / 3)]
            detections_filter = detections[valid_indices]  # ใช้ indexing เพื่อเลือกเฉพาะที่ต้องการ

            # Build labels with speed estimation
            labels = []
            for tracker_id, conf, xyxy in zip(detections_filter.tracker_id, detections_filter.confidence, detections_filter.xyxy):
                result_label = f"#ID{tracker_id} [{conf:.2f}]:"
                if len(self.coordinates[tracker_id]) < self.video_info.fps / 2:
                    labels.append(f"{result_label} Calculating...")
                    continue
                
                position_start = np.array(self.coordinates[tracker_id][-1])
                position_end = np.array(self.coordinates[tracker_id][0])
                distance = np.linalg.norm(position_start - position_end)
                time_elapsed = len(self.coordinates[tracker_id]) / self.video_info.fps
                speed = distance / time_elapsed * 3.6  # Convert m/s to km/h
                labels.append(f"{result_label} {speed:.2f} km/h")

            # Annotate frame
            frame = self.trace_annotator.annotate(scene=frame, detections=detections_filter)
            frame = self.box_annotator.annotate(scene=frame, detections=detections_filter)
            frame = self.label_annotator.annotate(
                scene=frame, detections=detections_filter, labels=labels
            )
        return frame

# Get Info
def get_video_info(source_video):
    VideoInfo = namedtuple("VideoInfo", ["resolution_wh", "fps"])
    cap = cv2.VideoCapture(source_video)
    # ดึงข้อมูลวิดีโอ
    resolution_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()  # ปิด video capture
    return VideoInfo(resolution_wh, fps)


if __name__ == "__main__":
    YOLO_CONFIG = {
        "yolo_model": "src/assets/models/ObjectDetection_ForkliftDetection_v1_2025_02_24.pt",
        "yolo_class": 0,
        "confidence_threshold": 0.3,
        "iou_threshold": 0.7,
        "polygon_points": [[745, 209], [963, 209], [1528, 1079], [304, 1079]],
        "target_width": 4.6,
        "target_height": 23.1}
     
    source_video = 1
    
    # ดึงข้อมูลวิดีโอ
    video_info = get_video_info(source_video)
    pr.yellow(video_info)  # แสดงข้อมูลวิดีโอ
    
    estimator = SpeedEstimator(video_info, **YOLO_CONFIG)
    
    cap = cv2.VideoCapture(source_video)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = estimator.process_frame(frame)
        
        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()
