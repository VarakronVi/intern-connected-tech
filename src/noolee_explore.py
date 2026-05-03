import sys, os
import cv2
import numpy as np
import json

# สี fallback เผื่อไม่มี COLOR_STYLE_RGB
BRAND = (0, 200, 255)
WHITE = (255, 255, 255)
ACCENT = (60, 180, 255)

def pr(msg): print(msg)

class ImagePolygonDrawer:
    def __init__(self, source: str, window_width: int = 1000):
        self.source_path = source
        self.img = cv2.imread(self.source_path)
        if self.img is None:
            print(f"❌ ไม่พบรูปภาพที่ {self.source_path}")
            sys.exit(1)

        self.h, self.w = self.img.shape[:2]
        self.aspect = self.w / self.h

        self.window_name = "Click to add points | s=save | z=undo | q=quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        window_height = int(window_width / self.aspect)
        cv2.resizeWindow(self.window_name, window_width, window_height)

        self.poly_points = []
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.poly_points.append((x, y))

    def display(self):
        while True:
            canvas = self.img.copy()
            # วาดจุดและเส้น
            for p in self.poly_points:
                cv2.circle(canvas, p, 5, WHITE, -1)
            if len(self.poly_points) > 1:
                cv2.polylines(canvas, [np.array(self.poly_points, np.int32)], False, BRAND, 2)
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('z') and self.poly_points:
                self.poly_points.pop()
            elif key == ord('s'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.poly_points


if __name__ == "__main__":
    print("🖼️ โปรแกรมวาดเส้นบนรูปภาพ")
    img_path = input("👉 ใส่ path ของรูปภาพ (เช่น /Users/noolee/Desktop/track.jpg): ").strip()

    if not os.path.exists(img_path):
        print("❌ ไฟล์ไม่พบ! ตรวจสอบ path อีกครั้ง")
        sys.exit(1)

    drawer = ImagePolygonDrawer(img_path)
    points = drawer.display()
    print("\n✅ จุดที่คุณวาดได้:")
    print(json.dumps(points, indent=2, ensure_ascii=False))
