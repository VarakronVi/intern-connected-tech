import customtkinter as ctk
import tkinter as tk

class PolygonDrawer(ctk.CTkToplevel):
    def __init__(self, 
                 parent, 
                 frame_size: tuple[int, int] = (600, 400), 
                 canvas_width: int = 600, 
                 canvas_height: int = 400) -> None:
        super().__init__(parent)  # กำหนด Parent เป็นหน้าต่างหลัก
        self.title("🖍️ Draw Area with Mouse")
        self.geometry(f"{frame_size[0]}x{frame_size[1]}")
        
        # สร้าง Canvas สำหรับวาดรูป
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg="white")
        self.canvas.pack(pady=10)

        self.polygon_points = []  # เก็บพิกัดจุดที่คลิก
        self.lines = []  # เก็บเส้นที่ลากไว้
        self.canvas.bind("<Button-1>", self.on_click)  # คลิกซ้ายเพื่อเพิ่มจุด
        
        # ปุ่ม Finish
        self.finish_button = ctk.CTkButton(self, text="Finish", command=self.draw_polygon, state="disabled")
        self.finish_button.pack(pady=5)

    def on_click(self, event):
        """เมื่อคลิกบน Canvas จะเพิ่มจุดและวาดเส้นต่อกัน"""
        x, y = event.x, event.y
        self.polygon_points.append((x, y))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")  # วาดจุด

        if len(self.polygon_points) > 1:
            x1, y1 = self.polygon_points[-2]
            x2, y2 = self.polygon_points[-1]
            line = self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
            self.lines.append(line)

        # ลบเส้นปิดก่อนหน้า และเพิ่มเส้นใหม่
        if len(self.polygon_points) > 2:
            if len(self.lines) >= len(self.polygon_points):
                self.canvas.delete(self.lines[-2])
                self.lines.pop()

            x1, y1 = self.polygon_points[-1]
            x2, y2 = self.polygon_points[0]
            line = self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, dash=(4, 2))
            self.lines.append(line)

        # เปิดปุ่ม Finish เมื่อมีจุดมากกว่า 2 จุด
        if len(self.polygon_points) > 2:
            self.finish_button.configure(state="normal")

    def draw_polygon(self):
        """วาด Polygon จากจุดที่เลือกไว้"""
        if len(self.polygon_points) > 2:
            points = [coord for point in self.polygon_points for coord in point]
            self.canvas.create_polygon(points, fill="", outline="blue", width=2)

            # ปิดปุ่ม Finish หลังจากวาดเสร็จ
            self.finish_button.configure(state="disabled")

            print("Polygon Points:", self.polygon_points)  # แสดงค่าพิกัด