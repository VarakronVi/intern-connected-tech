import sys, os; sys.path.append(os.getcwd()) if os.getcwd() not in sys.path else None

from utils import *
import cv2
import numpy as np
import sys
import argparse
import json

from src.ui.system_design import COLOR_STYLE_RGB

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="video file path")
    return parser.parse_args()


class PolygonDrawer:
    def __init__(self,
                 window_width: int = 800,
                 source: str = 0) -> None:
        self.source_path = source
        self.polygon_points = []
        self.cap = cv2.VideoCapture(self.source_path)
        self.window_name = "Click on the frame to define the polygon. Press 's' to save the polygon."

        if not self.cap.isOpened():
            pr.red(f"Error opening video file: {self.source_path}")
            sys.exit()

        # Define the show_frame function
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Calculate the new window size to maintain aspect ratio
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = frame_width / frame_height
        window_height = int(window_width / aspect_ratio)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        cv2.setMouseCallback(self.window_name, self.draw_polygon)
        self.final_polygon_drawn = False

    def draw_polygon(self, 
                     event: int, 
                     x: int, 
                     y: int, 
                     flags: int, 
                     param: int) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_points.append((x, y))
            if len(self.polygon_points) > 2:
                self.final_polygon_drawn = True

    def display_frame(self):
        pr.black("Click on the frame to define the polygon. Press 's' to save the polygon.")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            polygon_frame = frame.copy()
            if not ret:
                break

            for point in self.polygon_points:
                cv2.circle(polygon_frame, point, 5, COLOR_STYLE_RGB.border_white_color, -1)

            if len(self.polygon_points) > 1:
                if self.final_polygon_drawn:
                    cv2.polylines(polygon_frame, [np.array(self.polygon_points)], isClosed=True, color=COLOR_STYLE_RGB.border_brand_color, thickness=3)
                else:
                    cv2.polylines(polygon_frame, [np.array(self.polygon_points)], isClosed=False, color=COLOR_STYLE_RGB.border_brand_color, thickness=3)

            cv2.imshow(self.window_name, polygon_frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        
        pr.yellow(f"Image Size: {frame.shape}")
        self.cap.release()
        cv2.destroyAllWindows()

        if len(self.polygon_points) < 4:
            pr.red("Polygon needs at least 4 points")
            sys.exit()

        # polygon = np.array(self.polygon_points)
        return self.polygon_points
    
def main_polygon_draw(opt):
    """Main function."""
    # Initialize the PolygonDrawer class
    drawer = PolygonDrawer(**vars(opt))
    polygon = drawer.display_frame()
    return polygon

if __name__ == "__main__":
    print("Start drawing the polygon")
    opt = parse_opt()
    polygon = main_polygon_draw(opt)
    
    print(json.dumps(polygon))  # ✅ แปลงเป็น JSON และ print ออกไป)
    # pr.green(f"Polygon Points: {polygon}")