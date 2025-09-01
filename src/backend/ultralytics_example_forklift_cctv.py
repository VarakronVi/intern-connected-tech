import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Define the polygon region
SOURCE = np.array([[745, 209], [963, 209], [1528, 1079], [304, 1079]])

# Define target rectangle dimensions for view transformation
TARGET_WIDTH = 4.6
TARGET_HEIGHT = 23.1
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        # Convert points to float32 for cv2.getPerspectiveTransform
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform an array of points using the perspective transform.
        Each point is in (x, y) format.
        """
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Get metadata from the original video
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    # Load the YOLO model
    model = YOLO("src/assets/models/ObjectDetection_ForkliftDetection_v1_2025_02_24.pt.pt")

    # Initialize ByteTrack
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    # Calculate optimal annotation parameters based on video resolution
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Video frames generator
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    # Create PolygonZone and ViewTransformer
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Dictionary to store recent positions for each tracker
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Create an output video sink
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Predict with YOLO
            result = model(frame)[0]

            # Convert detections to supervision format
            detections = sv.Detections.from_ultralytics(result)

            # Filter out low-confidence detections
            detections = detections[detections.confidence > args.confidence_threshold]

            # Keep only detections within the polygon region
            detections = detections[polygon_zone.trigger(detections)]

            # Apply non-maximum suppression
            detections = detections.with_nms(threshold=args.iou_threshold)

            # Update ByteTrack trackers
            detections = byte_track.update_with_detections(detections=detections)

            # OPTIONAL: Tracking only for a specific ID (e.g., tracker_id == 2)
            indices = [i for i, tracker_id in enumerate(detections.tracker_id) if tracker_id == 2]

            if indices:
                detections = detections[indices]
                # Get bottom-center anchors and transform them
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_points = view_transformer.transform_points(points=points).astype(float)

                # Save recent positions for each tracker
                for tracker_id, (x, y) in zip(detections.tracker_id, transformed_points):
                    coordinates[tracker_id].append((x, y))

                # Build labels showing speed in km/h
                labels = []
                for tracker_id in detections.tracker_id:
                    # Check if there are enough points for a meaningful speed estimate
                    if len(coordinates[tracker_id]) < video_info.fps / 2:
                        labels.append(f"#{tracker_id}")
                        continue

                    # Calculate speed: Euclidean distance over time
                    position_start = np.array(coordinates[tracker_id][-1])
                    position_end = np.array(coordinates[tracker_id][0])
                    distance = np.linalg.norm(position_start - position_end)
                    time_elapsed = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time_elapsed * 3.6  # Convert m/s to km/h
                    labels.append(f"#ID:2 {speed:.2f} km/h")

                # Annotate frame with detections, boxes, labels, and trace lines
                annotated_frame = frame.copy()
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            else:
                # If no valid detections, simply work on a copy of the original frame.
                annotated_frame = frame.copy()

            # Draw the polygon region on every frame
            cv2.polylines(
                annotated_frame,
                [SOURCE.astype(np.int32)],  # Convert to integer format for drawing
                isClosed=True,              # Close the polygon
                color=(240, 240, 129),          # Green color in BGR
                thickness=thickness         # Optimal line thickness
            )

            # Write the annotated frame (with polygon) to the output video
            sink.write_frame(annotated_frame)
