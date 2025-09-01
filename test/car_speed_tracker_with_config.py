import asyncio
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import deque
import json
import yaml
import os
from pathlib import Path
from datetime import datetime
import logging

class EvidenceRecorder:
    """บันทึกวิดีโอหลักฐานเฉพาะช่วงที่ตรวจพบความเร็วเกิน"""
    
    def __init__(self, output_dir="evidence", record_duration=8):
        self.output_dir = output_dir
        self.record_duration = record_duration
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.active_recordings = {}
        self.recording_lock = threading.Lock()
        
        logging.basicConfig(
            filename=os.path.join(output_dir, 'speed_violations.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        print(f"Evidence recorder initialized - Output: {output_dir}")
    
    def start_recording(self, violation_info, frame, timestamp, tracks=None):
        """เริ่มบันทึกเมื่อพบการละเมิด"""
        vehicle_id = violation_info['vehicle_id']
        
        if vehicle_id in self.active_recordings:
            return None
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        speed = violation_info['speed']
        filename = f"violation_{timestamp_str}_ID{vehicle_id}_{speed:.0f}kmh.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 25.0
            frame_size = (frame.shape[1], frame.shape[0])
            
            writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size, True)
            
            if writer.isOpened():
                with self.recording_lock:
                    self.active_recordings[vehicle_id] = {
                        'writer': writer,
                        'filepath': filepath,
                        'start_time': timestamp,
                        'frames_written': 0,
                        'violation_info': violation_info
                    }
                
                overlay_frame = self._add_evidence_overlay(frame, violation_info, timestamp, tracks)
                writer.write(overlay_frame)
                self.active_recordings[vehicle_id]['frames_written'] = 1
                
                self._log_violation(violation_info, filepath)
                
                print(f"Started recording evidence: {filename}")
                return filepath
            else:
                print(f"Failed to create video writer for vehicle {vehicle_id}")
                return None
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            return None
    
    def add_frame(self, frame, timestamp, tracks):
        """เพิ่มเฟรมไปยังการบันทึกที่กำลังทำงาน"""
        with self.recording_lock:
            to_remove = []
            
            for vehicle_id, recording in self.active_recordings.items():
                try:
                    elapsed = timestamp - recording['start_time']
                    
                    if elapsed < self.record_duration:
                        overlay_frame = self._add_evidence_overlay(
                            frame, recording['violation_info'], timestamp, tracks
                        )
                        recording['writer'].write(overlay_frame)
                        recording['frames_written'] += 1
                    else:
                        recording['writer'].release()
                        
                        print(f"Evidence recording completed: {os.path.basename(recording['filepath'])}")
                        print(f"  Frames: {recording['frames_written']}, Duration: {elapsed:.1f}s")
                        
                        to_remove.append(vehicle_id)
                        
                except Exception as e:
                    print(f"Error writing frame for vehicle {vehicle_id}: {e}")
                    to_remove.append(vehicle_id)
            
            for vehicle_id in to_remove:
                if vehicle_id in self.active_recordings:
                    del self.active_recordings[vehicle_id]
    
    def _add_evidence_overlay(self, frame, violation_info, timestamp, tracks=None):
        """เพิ่ม overlay สำหรับหลักฐาน"""
        overlay_frame = frame.copy()
        
        if violation_info and tracks:
            vehicle_id = violation_info['vehicle_id']
            
            # เช็คว่ารถยังอยู่ในเฟรมปัจจุบันหรือไม่
            if vehicle_id in tracks and tracks[vehicle_id]['missed_frames'] == 0:
                current_track = tracks[vehicle_id]
                x1, y1, x2, y2 = current_track['detection'][:4]
                
                # ได้ความเร็วปัจจุบัน
                current_speed = self._get_current_speed(current_track)
                
                # วาดกรอบแดง
                cv2.rectangle(overlay_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                
                # ข้อความที่เปลี่ยนตามความเร็วปัจจุบัน
                speed_text = f"LIVE RECORDING: {current_speed:.1f} km/h"
                limit_text = f"Speed Limit: {violation_info['speed_limit']} km/h"
                id_text = f"Vehicle ID: {vehicle_id}"
                
                # กล่องข้อความ
                cv2.rectangle(overlay_frame, (int(x1), int(y1)-90), (int(x1)+400, int(y1)), (0, 0, 0), -1)
                
                # เปลี่ยนสีตามสถานะ
                speed_color = (0, 0, 255) if current_speed > violation_info['speed_limit'] else (255, 255, 0)
                
                cv2.putText(overlay_frame, speed_text, (int(x1)+5, int(y1)-65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
                cv2.putText(overlay_frame, limit_text, (int(x1)+5, int(y1)-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(overlay_frame, id_text, (int(x1)+5, int(y1)-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # ถ้ารถหายไป จะไม่วาดกรอบแดง (แก้ปัญหากรอบติดค้าง)
        
        # Timestamp
        datetime_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(overlay_frame, (overlay_frame.shape[1]-200, 10), 
                     (overlay_frame.shape[1]-10, 40), (0, 0, 0), -1)
        cv2.putText(overlay_frame, datetime_str, (overlay_frame.shape[1]-195, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def _get_current_speed(self, track):
        """คำนวณความเร็วปัจจุบันของรถ"""
        if track.get('stable_speed', 0) > 0:
            return track['stable_speed']
        
        speeds = track.get('speeds', [])
        if len(speeds) > 0:
            return speeds[-1]  # ใช้ความเร็วล่าสุด
        
        return 0
    
    def _log_violation(self, violation_info, filepath):
        """บันทึก log การละเมิด"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'vehicle_id': violation_info['vehicle_id'],
            'speed_kmh': round(violation_info['speed'], 1),
            'speed_limit': violation_info['speed_limit'],
            'over_limit': round(violation_info['speed'] - violation_info['speed_limit'], 1),
            'video_file': os.path.basename(filepath)
        }
        
        logging.info(f"VIOLATION: {json.dumps(log_entry)}")
        
        json_file = os.path.join(self.output_dir, 'violations.json')
        violations = []
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    violations = json.load(f)
            except:
                violations = []
        
        violations.append(log_entry)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(violations, f, indent=2, ensure_ascii=False)
    
    def get_active_count(self):
        """ได้จำนวนการบันทึกที่กำลังทำงาน"""
        with self.recording_lock:
            return len(self.active_recordings)
    
    def stop_all(self):
        """หยุดการบันทึกทั้งหมด"""
        with self.recording_lock:
            for vehicle_id, recording in self.active_recordings.items():
                try:
                    recording['writer'].release()
                    print(f"Stopped recording for vehicle {vehicle_id}")
                except:
                    pass
            self.active_recordings.clear()

class ConfigManager:
    """จัดการการอ่านและตรวจสอบไฟล์ config.yaml"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """โหลดไฟล์ config.yaml"""
        if not os.path.exists(self.config_path):
            print(f"Config file not found: {self.config_path}")
            print("Please create config.yaml file")
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            print(f"Config loaded successfully: {self.config_path}")
            return config
        except yaml.YAMLError as e:
            print(f"Invalid config.yaml format: {e}")
            raise
        except Exception as e:
            print(f"Error reading config file: {e}")
            raise
    
    def validate_config(self):
        """ตรวจสอบความถูกต้องของการตั้งค่า"""
        errors = []
        
        if 'camera' not in self.config:
            errors.append("Missing camera settings")
        else:
            if 'rtsp_url' not in self.config['camera']:
                errors.append("Missing rtsp_url in camera settings")
            elif not self.config['camera']['rtsp_url'].strip():
                errors.append("rtsp_url cannot be empty")
        
        if 'speed_limit' not in self.config:
            errors.append("Missing speed_limit settings")
        else:
            if 'max_speed_kmh' not in self.config['speed_limit']:
                errors.append("Missing max_speed_kmh in speed_limit settings")
            else:
                speed = self.config['speed_limit']['max_speed_kmh']
                if not isinstance(speed, (int, float)) or speed <= 0:
                    errors.append("max_speed_kmh must be a number greater than 0")
        
        if 'measurement_area' not in self.config:
            errors.append("Missing measurement_area settings")
        else:
            area = self.config['measurement_area']
            if 'width_meters' not in area:
                errors.append("Missing width_meters in measurement_area")
            elif not isinstance(area['width_meters'], (int, float)) or area['width_meters'] <= 0:
                errors.append("width_meters must be a number greater than 0")
            
            if 'height_meters' not in area:
                errors.append("Missing height_meters in measurement_area")
            elif not isinstance(area['height_meters'], (int, float)) or area['height_meters'] <= 0:
                errors.append("height_meters must be a number greater than 0")
        
        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Config validation failed")
        
        print("Configuration validated successfully")
    
    def get_camera_config(self):
        return self.config['camera']
    
    def get_speed_limit(self):
        return self.config['speed_limit']['max_speed_kmh']
    
    def get_measurement_area(self):
        area = self.config['measurement_area']
        return area['width_meters'], area['height_meters']
    
    def get_advanced_config(self):
        default_advanced = {
            'model_path': 'yolo11n.pt',
            'confidence_threshold': 0.4,
            'iou_threshold': 0.5,
            'max_missed_frames': 10,
            'buffer_size': 2
        }
        return self.config.get('advanced', default_advanced)
    
    def print_config_summary(self):
        print("Configuration Summary:")
        print("=" * 40)
        print(f"RTSP URL: {self.config['camera']['rtsp_url']}")
        print(f"Speed Limit: {self.get_speed_limit()} km/h")
        width, height = self.get_measurement_area()
        print(f"Area Size: {width} x {height} meters")
        print("=" * 40)

class RTSPCapture:
    """High-performance RTSP capture"""
    def __init__(self, rtsp_url, buffer_size=1):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.buffer_size = buffer_size
        self.frame_count = 0
        self.start_time = None
        self.fps = 30.0
        
    def start(self):
        if self.running:
            return
            
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot connect to RTSP stream: {self.rtsp_url}")
            
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps
            print(f"Stream FPS: {self.fps}")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"RTSP capture started: {self.rtsp_url}")
        
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                current_time = time.time()
                frame_timestamp = self.start_time + (self.frame_count / self.fps)
                
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_timestamp = frame_timestamp
                
                self.frame_count += 1
            else:
                time.sleep(0.01)
                
    def read(self):
        with self.lock:
            if hasattr(self, 'frame_timestamp'):
                return self.ret, self.frame, self.frame_timestamp
            return self.ret, self.frame, time.time()
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

class RegionDrawer:
    """Interactive region selection"""
    def __init__(self, rtsp_capture):
        self.rtsp_capture = rtsp_capture
        self.points = []
        self.window_name = "Select Measurement Area - Click 4 points clockwise"
        
    def draw_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")

    def select_region(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.draw_points)
        
        print("Getting frame from camera...")
        
        for _ in range(50):
            ret, frame, _ = self.rtsp_capture.read()
            if ret and frame is not None:
                break
            time.sleep(0.033)
        
        if not ret or frame is None:
            raise RuntimeError("Cannot get frame from camera")
            
        reference_frame = frame.copy()
        
        while True:
            display_frame = reference_frame.copy()
            
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(display_frame, 0.8, overlay, 0.2, 0, display_frame)
            
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 12, (255, 255, 255), 2)
                cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                cv2.circle(display_frame, point, 4, (255, 255, 255), -1)
                
                cv2.putText(display_frame, f"{i+1}", (point[0]+15, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if len(self.points) > 1:
                for i in range(len(self.points)):
                    next_i = (i + 1) % len(self.points) if len(self.points) == 4 else i + 1
                    if next_i < len(self.points):
                        cv2.line(display_frame, self.points[i], self.points[next_i], (0, 255, 255), 2)
            
            if len(self.points) < 4:
                instruction = f"Click point {len(self.points)+1} of 4"
                color = (255, 255, 0)
            else:
                instruction = "Press SPACE to continue, R to reset"
                color = (0, 255, 0)
            
            cv2.putText(display_frame, instruction, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            progress = len(self.points) / 4
            cv2.rectangle(display_frame, (20, 50), (220, 65), (100, 100, 100), -1)
            if progress > 0:
                cv2.rectangle(display_frame, (20, 50), (20 + int(200 * progress), 65), (0, 255, 0), -1)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and len(self.points) == 4:
                break
            elif key == ord('r') or key == ord('R'):
                self.points = []
                print("Points reset")
            elif key == 27:
                cv2.destroyAllWindows()
                raise SystemExit("User cancelled")
        
        cv2.destroyAllWindows()
        return self.points

class ModernUI:
    """Clean modern UI renderer"""
    
    @staticmethod
    def draw_hud_background(frame, width, height=90):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (30, 30, 30), -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        cv2.rectangle(frame, (0, 0), (width, height), (80, 80, 80), 1)
    
    @staticmethod
    def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2):
        corner_length = 20
        
        # Top-left
        cv2.line(frame, (int(x1), int(y1)), (int(x1 + corner_length), int(y1)), color, thickness)
        cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1 + corner_length)), color, thickness)
        
        # Top-right
        cv2.line(frame, (int(x2 - corner_length), int(y1)), (int(x2), int(y1)), color, thickness)
        cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1 + corner_length)), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (int(x1), int(y2 - corner_length)), (int(x1), int(y2)), color, thickness)
        cv2.line(frame, (int(x1), int(y2)), (int(x1 + corner_length), int(y2)), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (int(x2), int(y2 - corner_length)), (int(x2), int(y2)), color, thickness)
        cv2.line(frame, (int(x2 - corner_length), int(y2)), (int(x2), int(y2)), color, thickness)
    
    @staticmethod
    def draw_trajectory(frame, positions, H_matrix, color, max_points=8):
        if len(positions) < 2:
            return
        
        world_pts = np.array(positions[-max_points:], dtype=np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(world_pts, H_matrix)
        
        for i in range(1, len(img_pts)):
            alpha = i / len(img_pts)
            thickness = max(1, int(2 * alpha))
            fade_color = tuple(int(c * alpha) for c in color)
            
            pt1 = tuple(map(int, img_pts[i-1][0]))
            pt2 = tuple(map(int, img_pts[i][0]))
            cv2.line(frame, pt1, pt2, fade_color, thickness)

class CarSpeedTracker:
    """Car speed tracker with evidence recording and clean UI"""
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Load settings from config
        camera_config = self.config.get_camera_config()
        self.rtsp_url = camera_config['rtsp_url']
        self.speed_limit = self.config.get_speed_limit()
        self.real_width_m, self.real_height_m = self.config.get_measurement_area()
        
        advanced_config = self.config.get_advanced_config()
        self.model_path = advanced_config['model_path']
        self.confidence_threshold = advanced_config['confidence_threshold']
        self.iou_threshold = advanced_config['iou_threshold']
        self.max_missed_frames = advanced_config['max_missed_frames']
        buffer_size = advanced_config['buffer_size']
        
        self.class_names = {2: "Car"}
        
        # World coordinate system
        self.world_points = np.array([
            [0, 0],                           
            [self.real_width_m, 0],                
            [self.real_width_m, self.real_height_m],    
            [0, self.real_height_m]                
        ], dtype=np.float32)
        
        self.region_points = None
        self.H_img_to_world = None
        self.H_world_to_img = None
        
        # Load YOLO model
        self.model = YOLO(self.model_path)
        print(f"YOLO model loaded: {self.model_path}")
        
        # Tracking
        self.tracks = {}
        self.next_id = 0
        
        # UI and evidence recording
        self.ui = ModernUI()
        self.evidence_recorder = EvidenceRecorder()
        self.show_trajectories = True
        self.recorded_violations = set()
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_display_time = time.time()
        
        # Threading
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.running = False
        
        # RTSP Capture
        self.rtsp_capture = RTSPCapture(self.rtsp_url, buffer_size=buffer_size)
        
    def set_region_points(self, region_points):
        self.region_points = np.array(region_points, dtype=np.float32)
        self.H_img_to_world = cv2.getPerspectiveTransform(self.region_points, self.world_points)
        self.H_world_to_img = cv2.getPerspectiveTransform(self.world_points, self.region_points)
        print(f"Region configured: {self.real_width_m}x{self.real_height_m} meters")
        
    def pixel_to_world(self, points):
        if len(points) == 0:
            return np.array([])
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        world_coords = cv2.perspectiveTransform(points_array, self.H_img_to_world)
        return world_coords.reshape(-1, 2)
    
    def is_point_in_region(self, point):
        return cv2.pointPolygonTest(self.region_points, point, False) >= 0
    
    def calculate_speed_kmh(self, distance_m, time_s):
        if time_s <= 0:
            return 0
        return (distance_m / time_s) * 3.6
    
    def is_speeding(self, speed_kmh):
        return speed_kmh > self.speed_limit
    
    def get_display_speed(self, track):
        """Calculate display speed"""
        if track.get('stable_speed', 0) > 0:
            speed = track['stable_speed']
            return speed, "STABLE", self._get_speed_color(speed)
        
        speeds = track.get('speeds', [])
        if len(speeds) >= 3:
            speeds_array = np.array(speeds)
            median_speed = np.median(speeds_array)
            std_speed = np.std(speeds_array)
            
            valid_speeds = speeds_array[
                np.abs(speeds_array - median_speed) <= 1.5 * std_speed
            ]
            
            if len(valid_speeds) > 0:
                accurate_speed = np.mean(valid_speeds)
                if accurate_speed > 0:
                    return accurate_speed, "ACCURATE", self._get_speed_color(accurate_speed)
        
        elif len(speeds) > 0:
            avg_speed = np.mean(speeds)
            if avg_speed > 0:
                return avg_speed, "PARTIAL", self._get_speed_color(avg_speed)
        
        # Fallback calculation
        positions = track.get('position_history', [])
        timestamps = track.get('time_history', [])
        
        if len(positions) >= 2 and len(timestamps) >= 2:
            total_distance = 0
            for i in range(1, len(positions)):
                total_distance += np.linalg.norm(positions[i] - positions[i-1])
            
            total_time = timestamps[-1] - timestamps[0]
            
            if total_time > 0.2 and total_distance > 0.3:
                fallback_speed = self.calculate_speed_kmh(total_distance, total_time)
                if fallback_speed > 0 and fallback_speed < 200:
                    return fallback_speed, "ESTIMATED", self._get_speed_color(fallback_speed)
        
        if len(positions) >= 2:
            movement = np.linalg.norm(positions[-1] - positions[0])
            if movement > 0.3:
                return 0, "MOVING", (100, 150, 255)
        
        return 0, "DETECTING", (150, 150, 150)
    
    def _get_speed_color(self, speed):
        if self.is_speeding(speed):
            return (0, 50, 255)  # Red
        elif speed > self.speed_limit * 0.8:
            return (0, 150, 255)  # Orange
        else:
            return (50, 200, 50)  # Green
    
    def update_tracks(self, detections, frame_timestamp, frame):
        """Update tracking system and check for violations"""
        valid_detections = []
        valid_centers = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if self.is_point_in_region((center_x, center_y)):
                valid_detections.append(detection)
                valid_centers.append((center_x, center_y))
        
        if not valid_centers:
            for track in self.tracks.values():
                track['missed_frames'] += 1
            return
        
        world_centers = self.pixel_to_world(valid_centers)
        
        if not self.tracks:
            for i, (detection, world_pos) in enumerate(zip(valid_detections, world_centers)):
                x1, y1, x2, y2, conf, cls = detection
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'last_pos': world_pos,
                    'last_time': frame_timestamp,
                    'position_history': [world_pos],
                    'time_history': [frame_timestamp],
                    'speeds': [],
                    'avg_speed': 0,
                    'stable_speed': 0,
                    'missed_frames': 0,
                    'detection': detection,
                    'class_name': "Car",
                    'confidence': conf
                }
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            if len(track_ids) > 0 and len(world_centers) > 0:
                distances = np.zeros((len(track_ids), len(world_centers)))
                for i, track_id in enumerate(track_ids):
                    for j, world_pos in enumerate(world_centers):
                        distances[i, j] = np.linalg.norm(
                            self.tracks[track_id]['last_pos'] - world_pos
                        )
                
                row_ind, col_ind = linear_sum_assignment(distances)
                matched_tracks = set()
                matched_detections = set()
                
                for track_idx, det_idx in zip(row_ind, col_ind):
                    if distances[track_idx, det_idx] < 5.0:
                        track_id = track_ids[track_idx]
                        track = self.tracks[track_id]
                        new_world_pos = world_centers[det_idx]
                        
                        track['position_history'].append(new_world_pos)
                        track['time_history'].append(frame_timestamp)
                        
                        if len(track['position_history']) > 10:
                            track['position_history'] = track['position_history'][-10:]
                            track['time_history'] = track['time_history'][-10:]
                        
                        time_diff = frame_timestamp - track['last_time']
                        if time_diff > 0.1:
                            distance = np.linalg.norm(new_world_pos - track['last_pos'])
                            instant_speed = self.calculate_speed_kmh(distance, time_diff)
                            
                            if instant_speed < 200:
                                track['speeds'].append(instant_speed)
                                
                                if len(track['speeds']) > 10:
                                    track['speeds'] = track['speeds'][-10:]
                                
                                speeds_array = np.array(track['speeds'])
                                if len(speeds_array) >= 3:
                                    median_speed = np.median(speeds_array)
                                    std_speed = np.std(speeds_array)
                                    valid_speeds = speeds_array[
                                        np.abs(speeds_array - median_speed) <= 1.5 * std_speed
                                    ]
                                    
                                    if len(valid_speeds) > 0:
                                        track['avg_speed'] = np.mean(valid_speeds)
                                        alpha = 0.15
                                        if track['stable_speed'] == 0:
                                            track['stable_speed'] = track['avg_speed']
                                        else:
                                            track['stable_speed'] = (alpha * track['avg_speed'] + 
                                                                   (1 - alpha) * track['stable_speed'])
                        
                        track['last_pos'] = new_world_pos
                        track['last_time'] = frame_timestamp
                        track['missed_frames'] = 0
                        track['detection'] = valid_detections[det_idx]
                        track['confidence'] = valid_detections[det_idx][4]
                        
                        matched_tracks.add(track_id)
                        matched_detections.add(det_idx)
                
                for track_id in track_ids:
                    if track_id not in matched_tracks:
                        self.tracks[track_id]['missed_frames'] += 1
                
                for det_idx in range(len(valid_detections)):
                    if det_idx not in matched_detections:
                        x1, y1, x2, y2, conf, cls = valid_detections[det_idx]
                        world_pos = world_centers[det_idx]
                        
                        self.tracks[self.next_id] = {
                            'id': self.next_id,
                            'last_pos': world_pos,
                            'last_time': frame_timestamp,
                            'position_history': [world_pos],
                            'time_history': [frame_timestamp],
                            'speeds': [],
                            'avg_speed': 0,
                            'stable_speed': 0,
                            'missed_frames': 0,
                            'detection': valid_detections[det_idx],
                            'class_name': "Car",
                            'confidence': conf
                        }
                        self.next_id += 1
        
        # Check for speed violations and start recording
        for track in self.tracks.values():
            if track['missed_frames'] == 0:
                display_speed, status, _ = self.get_display_speed(track)
                
                if display_speed > 0 and self.is_speeding(display_speed):
                    violation_key = f"{track['id']}_{int(frame_timestamp/10)}"
                    
                    if violation_key not in self.recorded_violations:
                        violation_info = {
                            'vehicle_id': track['id'],
                            'speed': display_speed,
                            'speed_limit': self.speed_limit,
                            'detection': track['detection'],
                            'confidence': track['confidence'],
                            'status': status
                        }
                        
                        self.evidence_recorder.start_recording(violation_info, frame, frame_timestamp, self.tracks)
                        self.recorded_violations.add(violation_key)
                        
                        print(f"RECORDING STARTED IMMEDIATELY: Vehicle {track['id']} - {display_speed:.1f} km/h")
        
        to_remove = [tid for tid, track in self.tracks.items() 
                    if track['missed_frames'] > self.max_missed_frames]
        for tid in to_remove:
            del self.tracks[tid]
    
    def draw_results(self, frame):
        """Draw clean modern UI with enhanced speed display"""
        height, width = frame.shape[:2]
        
        # Draw measurement region
        if self.region_points is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.region_points.astype(int)], (100, 255, 100))
            cv2.addWeighted(frame, 0.95, overlay, 0.05, 0, frame)
            cv2.polylines(frame, [self.region_points.astype(int)], True, (0, 200, 255), 2)
            
            for i, point in enumerate(self.region_points):
                x, y = map(int, point)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)
        
        active_tracks = len([t for t in self.tracks.values() if t['missed_frames'] == 0])
        speeding_count = 0
        stable_count = 0
        
        # Draw vehicle information with enhanced speed display
        for track in self.tracks.values():
            if track['missed_frames'] == 0:
                detection = track['detection']
                x1, y1, x2, y2, conf, cls = detection
                
                display_speed, status, color = self.get_display_speed(track)
                is_speeding = display_speed > 0 and self.is_speeding(display_speed)
                
                if is_speeding:
                    speeding_count += 1
                if status == "STABLE":
                    stable_count += 1
                
                # Draw corner box
                thickness = 3 if is_speeding else 2
                self.ui.draw_corner_box(frame, x1, y1, x2, y2, color, thickness)
                
                # Center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
                cv2.circle(frame, (center_x, center_y), 2, color, -1)
                
                # Draw trajectory
                if self.show_trajectories:
                    positions = track.get('position_history', [])
                    self.ui.draw_trajectory(frame, positions, self.H_world_to_img, color)
                
                # Enhanced Speed Display
                if display_speed > 0:
                    speed_box_width = 160
                    speed_box_height = 60
                    speed_box_x = max(10, min(int(x1), width - speed_box_width - 10))
                    speed_box_y = max(10, int(y1) - speed_box_height - 15)
                    
                    speed_color = (0, 0, 255) if is_speeding else (0, 200, 50)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (speed_box_x, speed_box_y), 
                                 (speed_box_x + speed_box_width, speed_box_y + speed_box_height), 
                                 (20, 20, 20), -1)
                    cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
                    
                    border_thickness = 4 if is_speeding else 2
                    cv2.rectangle(frame, (speed_box_x, speed_box_y), 
                                 (speed_box_x + speed_box_width, speed_box_y + speed_box_height), 
                                 speed_color, border_thickness)
                    
                    speed_text = f"{display_speed:.1f}"
                    cv2.putText(frame, speed_text, (speed_box_x + 10, speed_box_y + 35), 
                               cv2.FONT_HERSHEY_DUPLEX, 1.4, speed_color, 3)
                    
                    cv2.putText(frame, "km/h", (speed_box_x + 100, speed_box_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    status_text = "SPEEDING!" if is_speeding else "NORMAL"
                    status_color = (0, 0, 255) if is_speeding else (0, 255, 0)
                    cv2.putText(frame, status_text, (speed_box_x + 10, speed_box_y + 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                    
                    cv2.putText(frame, f"Vehicle {track['id']}", (speed_box_x, speed_box_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    limit_text = f"Limit: {self.speed_limit}"
                    cv2.putText(frame, limit_text, (speed_box_x + 100, speed_box_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    if is_speeding:
                        flash_time = int(time.time() * 4) % 2
                        if flash_time == 0:
                            warning_overlay = frame.copy()
                            cv2.rectangle(warning_overlay, (speed_box_x - 5, speed_box_y - 5), 
                                         (speed_box_x + speed_box_width + 5, speed_box_y + speed_box_height + 5), 
                                         (0, 0, 255), -1)
                            cv2.addWeighted(frame, 0.9, warning_overlay, 0.1, 0, frame)
                            
                            cv2.rectangle(frame, (speed_box_x, speed_box_y), 
                                         (speed_box_x + speed_box_width, speed_box_y + speed_box_height), 
                                         (0, 0, 255), 4)
                
                else:
                    status_box_width = 140
                    status_box_height = 40
                    status_box_x = max(10, min(int(x1), width - status_box_width - 10))
                    status_box_y = max(10, int(y1) - status_box_height - 10)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (status_box_x, status_box_y), 
                                 (status_box_x + status_box_width, status_box_y + status_box_height), 
                                 (30, 30, 30), -1)
                    cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
                    
                    cv2.rectangle(frame, (status_box_x, status_box_y), 
                                 (status_box_x + status_box_width, status_box_y + status_box_height), 
                                 (100, 100, 100), 1)
                    
                    cv2.putText(frame, f"Vehicle {track['id']}", (status_box_x + 5, status_box_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.putText(frame, status, (status_box_x + 5, status_box_y + 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Calculate FPS
        current_time = time.time()
        if self.last_display_time > 0:
            display_fps = 1.0 / (current_time - self.last_display_time)
            self.fps_counter.append(display_fps)
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
        else:
            avg_fps = 0
        self.last_display_time = current_time
        
        # Enhanced main HUD
        self.ui.draw_hud_background(frame, width, 90)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Vehicles: {active_tracks}", (150, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Stable Tracking: {stable_count}", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Recording status
        active_recordings = self.evidence_recorder.get_active_count()
        if active_recordings > 0:
            cv2.putText(frame, f"LIVE RECORDING: {active_recordings}", (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if speeding_count > 0:
            flash_time = int(time.time() * 3) % 2
            violation_color = (0, 0, 255) if flash_time == 0 else (0, 50, 200)
            cv2.putText(frame, f"VIOLATIONS: {speeding_count}", (250, 55), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, violation_color, 3)
            
            if flash_time == 0:
                warning_overlay = frame.copy()
                cv2.rectangle(warning_overlay, (245, 35), (400, 65), (0, 0, 100), -1)
                cv2.addWeighted(frame, 0.9, warning_overlay, 0.1, 0, frame)
        else:
            cv2.putText(frame, "All vehicles within limit", (250, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Speed limit display
        limit_text = f"SPEED LIMIT: {self.speed_limit} km/h"
        text_size = cv2.getTextSize(limit_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        cv2.putText(frame, limit_text, (width - text_size[0] - 20, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        
        area_text = f"Area: {self.real_width_m:.1f}m x {self.real_height_m:.1f}m"
        text_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, area_text, (width - text_size[0] - 20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Violation summary
        if speeding_count > 0:
            summary_text = f"WARNING: {speeding_count} vehicle(s) exceeding speed limit - RECORDING LIVE"
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            banner_y = 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (width, banner_y + 25), (0, 0, 150), -1)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
            
            cv2.putText(frame, summary_text, ((width - text_size[0]) // 2, banner_y + 17), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Help controls
        help_texts = [
            "Q - Quit", "R - Reset", "T - Trajectories", "S - Statistics", "E - Evidence"
        ]
        
        for i, text in enumerate(help_texts):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            y_pos = height - 20 - (len(help_texts) - i - 1) * 18
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (width - text_size[0] - 15, y_pos - 12), 
                         (width - 5, y_pos + 3), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
            
            cv2.putText(frame, text, (width - text_size[0] - 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    async def capture_frames_async(self):
        while self.running:
            ret, frame, timestamp = self.rtsp_capture.read()
            if ret and frame is not None:
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    self.frame_queue.put_nowait((frame, timestamp))
                except:
                    pass
            await asyncio.sleep(0.01)
    
    async def process_frames_async(self):
        while self.running:
            try:
                frame, frame_timestamp = self.frame_queue.get_nowait()
                
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, 
                    lambda: self.model(frame, verbose=False, conf=self.confidence_threshold, 
                                     iou=self.iou_threshold, imgsz=640, classes=[2])
                )
                
                detections = []
                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == 2:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            detections.append([x1, y1, x2, y2, conf, cls])
                
                self.update_tracks(detections, frame_timestamp, frame)
                self.evidence_recorder.add_frame(frame, frame_timestamp, self.tracks)
                
                result_frame = self.draw_results(frame)
                
                try:
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except Empty:
                            pass
                    self.result_queue.put_nowait(result_frame)
                except:
                    pass
                        
            except Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def display_frames_async(self):
        while self.running:
            try:
                frame = self.result_queue.get_nowait()
                cv2.imshow("Car Speed Tracker with Live Evidence Recording", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    self.tracks = {}
                    self.recorded_violations.clear()
                    print("All tracks and violations reset")
                elif key == ord('t'):
                    self.show_trajectories = not self.show_trajectories
                    status = "enabled" if self.show_trajectories else "disabled"
                    print(f"Trajectory display {status}")
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('e'):
                    self.print_evidence_status()
                elif key == ord('c'):
                    self.config.print_config_summary()
                    
            except Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Display error: {e}")
                await asyncio.sleep(0.01)
    
    def print_statistics(self):
        print("\nLive Speed Monitoring Statistics")
        print("=" * 40)
        print(f"Speed Limit: {self.speed_limit} km/h")
        print(f"Area: {self.real_width_m:.1f}m x {self.real_height_m:.1f}m")
        print(f"Active Recordings: {self.evidence_recorder.get_active_count()}")
        print("-" * 40)
        
        active_cars = [t for t in self.tracks.values() if t['missed_frames'] == 0]
        if active_cars:
            for track in active_cars:
                display_speed, status, _ = self.get_display_speed(track)
                is_violation = display_speed > 0 and self.is_speeding(display_speed)
                
                violation_text = "VIOLATION" if is_violation else "NORMAL"
                
                print(f"Vehicle {track['id']}:")
                print(f"  Speed: {display_speed:.1f} km/h ({status})")
                print(f"  Status: {violation_text}")
                print(f"  Confidence: {track['confidence']:.2f}")
                print()
        else:
            print("No active vehicles detected")
        print("=" * 40)
    
    def print_evidence_status(self):
        print("\nLive Evidence Recording Status")
        print("=" * 40)
        print(f"Active recordings: {self.evidence_recorder.get_active_count()}")
        print(f"Total violations recorded: {len(self.recorded_violations)}")
        print(f"Output directory: {self.evidence_recorder.output_dir}")
        
        if os.path.exists(self.evidence_recorder.output_dir):
            video_files = [f for f in os.listdir(self.evidence_recorder.output_dir) 
                          if f.endswith('.mp4')]
            if video_files:
                print(f"Evidence files: {len(video_files)}")
                for f in sorted(video_files)[-5:]:
                    print(f"  {f}")
            else:
                print("No evidence files yet")
        print("=" * 40)
    
    async def run_async(self):
        print("Starting Enhanced Car Speed Tracker with Live Evidence Recording")
        print(f"Speed Limit: {self.speed_limit} km/h")
        print(f"Area: {self.real_width_m} x {self.real_height_m} meters")
        print(f"Evidence Recording: {self.evidence_recorder.output_dir}")
        print("\nControls:")
        print("  Q - Quit")
        print("  R - Reset tracks") 
        print("  S - Show statistics")
        print("  T - Toggle trajectories")
        print("  E - Evidence status")
        print("  C - Show configuration")
        print("\nFeatures:")
        print("  ✓ Live speed tracking with real-time updates")
        print("  ✓ Immediate recording when violations detected")
        print("  ✓ Dynamic overlay that follows vehicle movement")
        print("  ✓ Auto-disappearing frames when vehicles leave area")
        
        self.rtsp_capture.start()
        time.sleep(3)
        
        self.running = True
        
        try:
            await asyncio.gather(
                self.capture_frames_async(),
                self.process_frames_async(),
                self.display_frames_async()
            )
        except KeyboardInterrupt:
            print("Stopping system...")
        finally:
            self.running = False
            self.evidence_recorder.stop_all()
            self.rtsp_capture.stop()
            cv2.destroyAllWindows()
            print("System stopped")

async def main():
    print("CAR SPEED TRACKER WITH LIVE EVIDENCE RECORDING")
    print("Configurable via config.yaml")
    print("Auto-records evidence with live speed tracking")
    print()
    
    try:
        print("Loading configuration...")
        config_manager = ConfigManager("config.yaml")
        config_manager.validate_config()
        config_manager.print_config_summary()
        
        print("Connecting to camera...")
        camera_config = config_manager.get_camera_config()
        temp_capture = RTSPCapture(camera_config['rtsp_url'])
        temp_capture.start()
        time.sleep(3)
        
        print("Select measurement region...")
        drawer = RegionDrawer(temp_capture)
        region_points = drawer.select_region()
        temp_capture.stop()
        
        print("Region selected:", region_points)
        
        tracker = CarSpeedTracker(config_manager)
        tracker.set_region_points(region_points)
        
        print("Starting tracking with live evidence recording...")
        await tracker.run_async()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nCreate config.yaml file with this template:")
        print("-" * 40)
        print("""camera:
  rtsp_url: "rtsp://localhost:8554/stream1"

speed_limit:
  max_speed_kmh: 15

measurement_area:
  width_meters: 10.0
  height_meters: 5.0

advanced:
  model_path: "yolo11n.pt"
  confidence_threshold: 0.4
  iou_threshold: 0.5
  max_missed_frames: 10
  buffer_size: 2""")
        print("-" * 40)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("PyYAML library not found")
        print("Install with: pip install PyYAML")
        exit(1)
    
    asyncio.run(main())