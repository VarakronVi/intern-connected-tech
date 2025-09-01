import customtkinter as ctk
import os
import cv2
from PIL import Image, ImageTk
import threading
import multiprocessing
import numpy as np
from collections import namedtuple
import subprocess
import json

from src.ui.system_design import SYSTEM_DESIGN, COLOR_STYLE, COLOR_STYLE_BGR
from src.ui.polygon_draw import PolygonDrawer
from src.backend.car_speed import SpeedEstimator
from src.get_polygon import main_polygon_draw
from utils import *

class App(ctk.CTk):
    def __init__(self,
                 frame_size: tuple[int, int] = (1920, 1080),
                 config_models: dict = {"Forklift": ("model_forklift.pt", 0),
                                  "Car": ("model_car.pt", 2)}) -> None:
        # Setting the theme
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("🚗 🏁 Speed Estimation Application")
        self.geometry(f"{frame_size[0]}x{frame_size[1]}")
        
        # Initialize all the variables
        self.frame_size = frame_size
        self.rtsp_path = ""
        ## Camera
        self.camera_width = 0
        self.camera_height = 0
        self.canvas_width = 0
        self.canvas_height = 0
        self.fps = 0
        ## Model
        self.polygon_points = [(213, 1051), (1495, 1042), (955, 207), (748, 205)]
        self.polygon_width = 4.6
        self.polygon_height = 23.1
        self.conf = 0.2
        self.iou = 0.8
        self.config_models = config_models
        self.model_name = list(self.config_models.keys())
        self.YOLO_CONFIG = {}
        ## Additional
        self.capture = None
        self.running = False
        self.flip = False
        ## Mutiprocessing
        self.running = multiprocessing.Value('b', False)  # ใช้ shared memory สำหรับตรวจสอบ state
        self.frame_queue = multiprocessing.Queue()  # ใช้ Queue ข้าม Process
        self.annotation_frame_queue = multiprocessing.Queue()
        
        # Start application
        self.setup_ui()
        
        # Setup Defult Model
        self.get_model_options(self.model_name[0])
        
        
    def setup_ui(self):
        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images
        self.image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "images")
        self.ct_logo_image = ctk.CTkImage(Image.open(os.path.join(self.image_path, "CT_logo_full.png")), size=(192, 69))
        self.home_image = ctk.CTkImage(light_image=Image.open(os.path.join(self.image_path, "home.png")), size=(20, 20))
        self.setting = ctk.CTkImage(light_image=Image.open(os.path.join(self.image_path, "setting.png")), size=(20, 20))
        self.manage_area = ctk.CTkImage(light_image=Image.open(os.path.join(self.image_path, "scalability.png")), size=(15, 15))
        
        # Path
        self.area_setting_path = os.path.join(self.image_path, "area_setting.png")

        # create navigation frame
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_STYLE.bg_gray_light_color)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)
        
        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="", image=self.ct_logo_image, compound="left")
        self.navigation_frame_label.grid(row=0, column=0, padx=1, pady=1)

        self.home_button = ctk.CTkButton(self.navigation_frame, 
                                         corner_radius=0, height=40, border_spacing=10, 
                                         text="Home", image=self.home_image, anchor="w", 
                                         command=self.home_button_event,
                                         **SYSTEM_DESIGN["Navigation"])
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.setting_button = ctk.CTkButton(self.navigation_frame, 
                                            corner_radius=0, height=40, border_spacing=10, 
                                            text="Setting", image=self.setting, anchor="w", 
                                            command=self.setting_button_event,
                                            **SYSTEM_DESIGN["Navigation"])
        self.setting_button.grid(row=2, column=0, sticky="ew")

        # create window
        self.open_home_window()
        self.open_setting_window()

        # select default frame
        self.select_frame_by_name("home")
    
    #--------------------------
    # Navigation Botton Functions
    #--------------------------
    def select_frame_by_name(self, name):
        # set button color for selected button
        if name == "home":
            self.home_button.configure(**SYSTEM_DESIGN["Navigation-Clicked"])
        else:
            self.home_button.configure(**SYSTEM_DESIGN["Navigation"])
        if name == "setting":
            self.setting_button.configure(**SYSTEM_DESIGN["Navigation-Clicked"])
        else: 
            self.setting_button.configure(**SYSTEM_DESIGN["Navigation"])

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
            
        if name == "setting":
            self.setting_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.setting_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def setting_button_event(self):
        self.select_frame_by_name("setting")
    
    #--------------------------
    # Botton Functions
    #--------------------------
    def toggle_flip(self):
        self.flip = not self.flip
        
    def start_stream(self):
        if not self.rtsp_path:
            print("RTSP path is not set.")
            self.screen_with_message(self.canvas_realtime, "Please Setting RTSP")
            return
        
        ## Spped Set Valiable
        self.YOLO_CONFIG = {
            "yolo_model": self.model,
            "yolo_class": int(self.class_id),
            "confidence_threshold": self.conf,
            "iou_threshold": self.iou,
            "polygon_points": self.polygon_points,
            "target_width": self.polygon_width,
            "target_height": self.polygon_height}

        self.running.value = True
        self.start_button.configure(**SYSTEM_DESIGN["Button-Disabled"])
        self.stop_button.configure(**SYSTEM_DESIGN["Button-Secondary"])
        self.class_option.configure(state="disabled")
        
        # เริ่ม Thread และ Process
        t1 = threading.Thread(target=self.update_frame, args=(self.canvas_realtime,), daemon=True)
        resolution_wh = (self.camera_width, self.camera_height)
        p = multiprocessing.Process(target=get_speed, args=(self.running, self.frame_queue, self.annotation_frame_queue, self.YOLO_CONFIG, resolution_wh, self.fps), daemon=True)
        t3 = threading.Thread(target=self.display_frame, args=(self.canvas_realtime,), daemon=True)

        t1.start()
        p.start()
        t3.start()

        self.process = p  # เก็บอ้างอิงไว้เพื่อปิด Process ทีหลัง
    
    def stop_stream(self):
        self.running.value = False  # สั่งให้ทุก Process & Thread หยุดทำงาน
        # ปิด Process อย่างปลอดภัย
        if hasattr(self, "process") and self.process.is_alive():
            print("Stopping process...")
            self.process.terminate()  # Kill process ทันที
            self.process.join(timeout=2)  # รอ process จบ ไม่เกิน 2 วินาที
            print("Process stopped.")
        
        self.start_button.configure(**SYSTEM_DESIGN["Button-Primary"])
        self.stop_button.configure(**SYSTEM_DESIGN["Button-Disabled"])
        self.class_option.configure(state="normal")
        self.show_black_screen(self.canvas_realtime)
        
    def get_model_options(self, selected_value):
        """ Callback function ที่ทำงานเมื่อมีการเลือกค่าใน OptionMenu """
        if selected_value in self.config_models:
            self.model, self.class_id = self.config_models[selected_value]
            
            conf_value = self.model_conf_info.get()
            iou_value = self.model_iou_info.get()
            if conf_value and iou_value:
                self.conf, self.iou = float(conf_value), float(iou_value)  # แปลงเป็น float หากต้องการ
            
            w_value = self.area_w_info_field.get()
            h_value = self.area_h_info_field.get()
            if conf_value and iou_value:
                self.polygon_width, self.polygon_height = float(w_value), float(h_value)  # แปลงเป็น float หากต้องการ
        else:
            pr.yellow(f"{selected_value} not found in data.")
    
    #--------------------------
    # Helper Functions
    #--------------------------
    def create_breadcrumbs(self, frame, path_list, total_columns=3):
        # Create a frame breadcrumbs
        parent = ctk.CTkFrame(frame, fg_color="transparent")
        parent.grid(row=0, column=0, padx=(20, 20), pady=10, sticky="ew")
        
        # Add Line 
        bottom_border = ctk.CTkFrame(frame, height=1, fg_color=COLOR_STYLE.border_gray_light_color)
        bottom_border.grid(row=0, column=0, columnspan=total_columns, padx=0, pady=(50, 0), sticky="ew")
        
        """ ฟังก์ชันสร้าง Breadcrumbs """
        for i, text in enumerate(path_list):
            if i < len(path_list) - 1:
                label = ctk.CTkLabel(parent, text=text, text_color=COLOR_STYLE.text_gray_light_color)
                label.pack(side="left", padx=5)
                arrow = ctk.CTkLabel(parent, text="/", text_color=COLOR_STYLE.border_gray_light_color)
                arrow.pack(side="left", padx=5)
            else:
                label = ctk.CTkLabel(parent, text=text, text_color=COLOR_STYLE.text_brand_color)
                label.pack(side="left", padx=5)
    
    def update_entry(self, parent, new_value):
        """ ฟังก์ชันอัปเดตค่าที่แสดงใน CTkEntry """
        parent.configure(state="normal")  # ปลดล็อกก่อนเปลี่ยนค่า
        parent.delete(0, "end")  # ล้างค่าเก่า
        parent.insert(0, new_value)  # ใส่ค่าจากพารามิเตอร์
        parent.configure(state="disabled")  # ล็อกอีกครั้งให้แก้ไม่ได้
    
    def show_black_screen(self, parent):
        self.update_canvas_size(parent)
        black_frame = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        image = Image.fromarray(black_frame)
        imgtk = ImageTk.PhotoImage(image=image)
        parent.create_image(0, 0, anchor="nw", image=imgtk)
        parent.imgtk = imgtk
    
    def screen_with_message(self, parent, message="Please Setting RTSP"):
        self.update_canvas_size(parent)
        parent.create_text(self.canvas_width/2, self.canvas_height/2, 
                           text=message, 
                           font=("Arial", 24), 
                           fill=COLOR_STYLE.text_brand_color)
    
    def update_canvas_size(self, parent):
        # ดึงขนาดปัจจุบันของ canvas
        parent.update_idletasks() # อัปเดต layout GUI ก่อนอ่านค่าขนาด
        self.canvas_width = parent.winfo_width()
        self.canvas_height = parent.winfo_height()
        # ป้องกันการหารด้วย 0 กรณีที่ canvas ยังไม่มีขนาด
        if self.canvas_width <= 1 or self.canvas_height <= 1:
            self.canvas_width = 900
            self.canvas_height = 480
        #     pr.red(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
        # else:
        #     pr.green(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
    
    def auto_rescale_frame(self, parent, frame):
        if self.flip:
            frame = cv2.flip(frame, 1)
        self.camera_height, self.camera_width, _ = frame.shape  # ขนาดภาพต้นฉบับ
        # ดึงขนาดปัจจุบันของ canvas
        self.update_canvas_size(parent)
        # คำนวณอัตราส่วนที่เหมาะสม
        scale = (self.canvas_width / self.camera_width)
        new_w = self.canvas_width # fix width
        new_h = int(self.camera_height * scale)
        # ปรับขนาดโดยรักษาอัตราส่วน
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # อัปเดตขนาดของ canvas ตามขนาดภาพ
        parent.configure(width=new_w, height=new_h)
        return resized_frame
    
    def display_image(self, parent, img_path):
        try:
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            frame = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            resized_frame = self.auto_rescale_frame(parent, frame)
            image = Image.fromarray(resized_frame)
            imgtk2 = ImageTk.PhotoImage(image=image)  # 🔹 เก็บอ้างอิงใน self
            parent.create_image(0, 0, anchor="nw", image=imgtk2)
            parent.imgtk = imgtk2  # ป้องกันการถูกล้างหน่วยความจำ
        except Exception as e:
            print(f"❌ Error loading image: {e}")
    
    def display_frame(self, parent):
        while self.running.value:
            if not self.annotation_frame_queue.empty():
                frame = self.annotation_frame_queue.get()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = self.auto_rescale_frame(parent, frame)
                image = Image.fromarray(resized_frame)
                imgtk = ImageTk.PhotoImage(image=image)
                parent.create_image(0, 0, anchor="nw", image=imgtk)
                parent.imgtk = imgtk  # ป้องกัน garbage collection

    #--------------------------
    # CCTV Functions
    #--------------------------
    def capture_image(self):
        cap = cv2.VideoCapture(self.rtsp_path)
        if not cap.isOpened():
            print(f"Cannot open stream from {self.rtsp_path}")
            return
        # Check black frame from video: normaly 1-2 frames are black
        def is_not_black_frame(frame):
            return np.any(frame > 0)
        # Get FPS
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if is_not_black_frame(frame):
                cv2.imwrite(self.area_setting_path, frame)
                print("Image saved.")
                cap.release()
                break
    
    def update_frame(self, parent):
        self.capture = cv2.VideoCapture(self.rtsp_path)
        if not self.capture.isOpened():
            print(f"Failed to open stream: {self.rtsp_path}")
            self.show_black_screen(parent)
            return
        while self.running.value:
            ret, frame = self.capture.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)  # ส่งภาพไปที่ Queue
        self.capture.release()
        self.show_black_screen(parent)
    
    def draw_polygon_on_image(self, polygon_points):
        image = cv2.imread(self.area_setting_path)
        self.SOURCE = np.array(polygon_points)
        cv2.polylines(
            image,
            [self.SOURCE.astype(np.int32)],
            isClosed=True,
            color=COLOR_STYLE_BGR.border_brand_color,
            thickness=3
        )
        cv2.imwrite(self.area_setting_path, image)
    
    def try_connect_camera(self):
        self.rtsp_path = self.rtsp_path_field.get()
        if self.rtsp_path == "" or self.rtsp_path == None:
            print("RTSP path is empty.")
            return
        # If Used Local Camera: Change rtsp_path to <int>
        if self.rtsp_path.isdigit():
            self.rtsp_path = int(self.rtsp_path)
        # Capture
        self.capture_image()
        # Show Image
        self.display_image(self.canvas_image, self.area_setting_path)
    
    #--------------------------
    # Window
    #--------------------------
    def open_home_window(self):
        # Create home frame
        self.home_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_STYLE.bg_white_color)
        self.home_frame.grid_columnconfigure(0, weight=1)  # ให้ Column 0 ขยายเต็มแนวกว้าง
        self.home_frame.grid_rowconfigure(1, weight=1)  # ให้ Row 1 ขยายเต็มแนวสูง
        
        # Get Breadcrumbs Frame
        self.create_breadcrumbs(self.home_frame, ["Speed Estimation", "Home"])
        
        # Create Canvas frame
        self.canvas_home_frame = ctk.CTkFrame(self.home_frame, corner_radius=0, fg_color=COLOR_STYLE.bg_white_color)
        self.canvas_home_frame.grid(row=1, column=0, padx=(20, 20), sticky="nsew")
        self.canvas_home_frame.grid_columnconfigure(0, weight=1)  # ขยาย column ซ้าย
        self.canvas_home_frame.grid_rowconfigure(0, weight=1)  # ขยาย row 1

        ## Canvas
        self.canvas_realtime = ctk.CTkCanvas(self.canvas_home_frame, 
                                    bg=COLOR_STYLE.bg_white_color)
        self.canvas_realtime.grid(row=0, column=0, pady=10, sticky="nsew")

        ## Button Frame
        self.button_frame = ctk.CTkFrame(self.canvas_home_frame, fg_color="transparent")
        self.button_frame.grid(row=1, column=0, pady=(0, 15), sticky="nsew")  # ชิดขวาล่าง

        ## Drowdown Class
        self.class_option = ctk.CTkComboBox(
            self.button_frame, 
            values=self.model_name,
            command=self.get_model_options,
            **SYSTEM_DESIGN["Dropdown"]
        )
        self.class_option.pack(side="left", padx=5)

        ### Stop Button
        self.stop_button = ctk.CTkButton(
            self.button_frame, 
            text="Stop", 
            command=self.stop_stream, 
            **SYSTEM_DESIGN["Button-Disabled"])
        self.stop_button.pack(side="right", padx=5)  # อยู่ติดกับปุ่ม Start
        
        ### Start Button
        self.start_button = ctk.CTkButton(
            self.button_frame, 
            text="Start", 
            command=self.start_stream,
            **SYSTEM_DESIGN["Button-Primary"]
        )
        self.start_button.pack(side="right", padx=5)  # วางเรียงกันแนวนอน

    def open_setting_window(self):
        # create setting frame
        self.setting_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_STYLE.bg_white_color)
        self.setting_frame.grid_columnconfigure(0, weight=1)
        self.setting_frame.grid_rowconfigure(2, weight=1)  # ให้ Row 1 ขยายเต็มแนวสูง
        
        # Get Breadcrumbs Frame
        self.create_breadcrumbs(self.setting_frame, ["Speed Estimation", "Setting"])
        
        # Create RTSP Path Frame
        self.camera_rtsp_frame = ctk.CTkFrame(self.setting_frame, corner_radius=0, fg_color=COLOR_STYLE.bg_white_color)
        self.camera_rtsp_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 20), pady=(20,10))
        self.camera_rtsp_frame.grid_columnconfigure(0, weight=1)
        
        ## Get RTSP Path
        rtsp_label = ctk.CTkLabel(self.camera_rtsp_frame, 
                                         text="RTSP Path",
                                         **SYSTEM_DESIGN["Label"])
        rtsp_label.pack(side="left", padx=5)
        self.rtsp_path_field = ctk.CTkEntry(self.camera_rtsp_frame, 
                                        placeholder_text="rtsp://admin:12345678@192.168.254.101:554/Streaming/channels/101",
                                        **SYSTEM_DESIGN["Input-Field"])
        self.rtsp_path_field.pack(side="left", padx=5, expand=True, fill="x")
        
        ## Connecting Button
        self.connecting_camera_button = ctk.CTkButton(
            self.camera_rtsp_frame, 
            text="Connect", 
            command=self.try_connect_camera,
            **SYSTEM_DESIGN["Button-Primary"]
        )
        self.connecting_camera_button.pack(side="left", padx=5)  # วางเรียงกันแนวนอน
        
        # Canvas
        self.canvas_image = ctk.CTkCanvas(self.setting_frame,
                                    bg=COLOR_STYLE.bg_white_color)
        self.canvas_image.grid(row=2, column=0, padx=(20, 10), pady=(0, 10), sticky="nsew")
        self.display_image(self.canvas_image, self.area_setting_path)
        
        # Create Show Info. Frame
        self.camera_info_frame = ctk.CTkFrame(self.setting_frame,
                                              **SYSTEM_DESIGN["Frame-Info"])
        self.camera_info_frame.grid(row=2, column=1, padx=(10, 20), pady=(0, 10), sticky="nsew")
        
        ## Camera Info.
        camera_info_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="📷 Camera Information",
                                         **SYSTEM_DESIGN["Label-Bold"])
        camera_info_label.grid(row=0, column=0, columnspan=2, padx=(10, 20), pady=(20, 10), sticky="w")
        ### Width px
        camera_w_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="Width: ",
                                         **SYSTEM_DESIGN["Label"])
        camera_w_label.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        camera_w_info = ctk.CTkEntry(self.camera_info_frame,
                                    **SYSTEM_DESIGN["Input-Field-Disabled"])
        self.update_entry(camera_w_info, self.camera_width)
        camera_w_info.grid(row=1, column=1, pady=(10, 10), sticky="w")
        unit_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="px",
                                         **SYSTEM_DESIGN["Label"])
        unit_label.grid(row=1, column=2, padx=(10, 20), pady=(10, 10), sticky="w")
        ### Hight px
        camera_h_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="Hight: ",
                                         **SYSTEM_DESIGN["Label"])
        camera_h_label.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        camera_h_info = ctk.CTkEntry(self.camera_info_frame,
                                    **SYSTEM_DESIGN["Input-Field-Disabled"])
        self.update_entry(camera_h_info, self.camera_height)
        camera_h_info.grid(row=2, column=1, pady=(10, 10), sticky="w")
        unit_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="px",
                                         **SYSTEM_DESIGN["Label"])
        unit_label.grid(row=2, column=2, padx=(10, 20), pady=(10, 10), sticky="w")
        ### FPS
        fps_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="FPS: ",
                                         **SYSTEM_DESIGN["Label"])
        fps_label.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        fps_info = ctk.CTkEntry(self.camera_info_frame,
                                    **SYSTEM_DESIGN["Input-Field-Disabled"])
        self.update_entry(fps_info, self.fps)
        fps_info.grid(row=3, column=1, pady=(10, 10), sticky="w")
        ### Filp image
        filp_image_switch = ctk.CTkSwitch(self.camera_info_frame, 
                                          text="Filp Image", 
                                          command=self.toggle_flip)
        filp_image_switch.grid(row=4, column=0, columnspan=2, padx=(20, 20), pady=(10, 10), sticky="w")
        
        ## Area Info.
        area_info_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="📐 Area Information",
                                         **SYSTEM_DESIGN["Label-Bold"])
        area_info_label.grid(row=5, column=0, columnspan=2, padx=(20, 20), pady=(30, 10), sticky="w")
        ### Width Area m
        area_w_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="Width Area: ",
                                         **SYSTEM_DESIGN["Label"])
        area_w_label.grid(row=6, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        self.area_w_info_field = ctk.CTkEntry(self.camera_info_frame,
                                               placeholder_text=self.polygon_width,
                                               **SYSTEM_DESIGN["Input-Field"])
        self.area_w_info_field.grid(row=6, column=1, pady=(10, 10), sticky="w")
        unit_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="m",
                                         **SYSTEM_DESIGN["Label"])
        unit_label.grid(row=6, column=2, padx=(10, 20), pady=(10, 10), sticky="w")
        ### Hight Area m
        area_h_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="Hight Area: ",
                                         **SYSTEM_DESIGN["Label"])
        area_h_label.grid(row=7, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        self.area_h_info_field = ctk.CTkEntry(self.camera_info_frame,
                                              placeholder_text=self.polygon_height,
                                              **SYSTEM_DESIGN["Input-Field"])
        self.area_h_info_field.grid(row=7, column=1, pady=(10, 10), sticky="w")
        unit_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="m",
                                         **SYSTEM_DESIGN["Label"])
        unit_label.grid(row=7, column=2, padx=(10, 20), pady=(10, 10), sticky="w")
        ## Model Info
        model_info_label = ctk.CTkLabel(self.camera_info_frame, 
                                         text="✨ Model Information",
                                         **SYSTEM_DESIGN["Label-Bold"])
        model_info_label.grid(row=8, column=0, columnspan=2, padx=(20, 20), pady=(30, 10), sticky="w")
        ### Model Conf Field
        model_conf_label = ctk.CTkLabel(self.camera_info_frame,
                                     text="Confidence Threshold: ",
                                     **SYSTEM_DESIGN["Label"])
        model_conf_label.grid(row=9, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        self.model_conf_info = ctk.CTkEntry(self.camera_info_frame,
                                            placeholder_text=self.conf,
                                            **SYSTEM_DESIGN["Input-Field"])
        self.model_conf_info.grid(row=9, column=1, pady=(10, 10), sticky="w")
        ### Model IOU Field
        model_iou_label = ctk.CTkLabel(self.camera_info_frame,
                                     text="IOU Threshold: ",
                                     **SYSTEM_DESIGN["Label"])
        model_iou_label.grid(row=10, column=0, padx=(20, 10), pady=(10, 10), sticky="w")
        self.model_iou_info = ctk.CTkEntry(self.camera_info_frame,
                                            placeholder_text=self.iou,
                                            **SYSTEM_DESIGN["Input-Field"])
        self.model_iou_info.grid(row=10, column=1, pady=(10, 10), sticky="w")
        
        # Manage Area Button
        self.manage_area_button = ctk.CTkButton(
                                    self.setting_frame, text="Manage Area", image=self.manage_area,
                                    command=self.open_polygon_drawer,
                                    **SYSTEM_DESIGN["Button-Primary"])
        self.manage_area_button.grid(row=3, column=0, padx=(20, 20), pady=(10, 20), sticky="w")
    
    # def open_polygon_drawer(self):
    #     """เปิดหน้าต่างวาดรูปหลายเหลี่ยม"""
    #     self.polygon_window = PolygonDrawer(self, 
    #                                         self.frame_size,
    #                                         self.canvas_width,
    #                                         self.canvas_height)
    #     self.polygon_window.grab_set()  #ให้หน้าต่างใหม่อยู่ด้านหน้า
    
    def open_polygon_drawer(self):
        """เปิดหน้าต่างวาดรูปหลายเหลี่ยม และรัน get_polygon.py"""
        try:
            # Run
            cmd = ["python", "src/get_polygon.py", "--source", str(self.rtsp_path)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Get
            output, error = process.communicate()
            if process.returncode == 0:
                lines = output.strip().split("\n")  # ✅ แยกบรรทัดที่ถูก print ออกมา
                json_output = lines[-1]  # ✅ ดึงบรรทัดสุดท้ายเป็น JSON
                if json_output:
                    polygon_list = json.loads(json_output)  # ✅ แปลง JSON เป็น list
                    self.polygon_points = [tuple(point) for point in polygon_list]  # ✅ แปลง list เป็น tuple
                    pr.green(f"✅ Polygon from script: {self.polygon_points}")
            # Save
            self.draw_polygon_on_image(self.polygon_points)
            # Show
            self.display_image(self.canvas_image, self.area_setting_path)

        except Exception as e:
            print(f"❌ Error running get_polygon.py: {e}")

#--------------------------
# Speed Estimation Call Function
#--------------------------
def get_speed(running, frame_queue, annotation_frame_queue, YOLO_CONFIG, resolution_wh, fps):
    """ ฟังก์ชันประมวลผลความเร็ว (ใช้ Multiprocessing) """
    VideoInfo = namedtuple("VideoInfo", ["resolution_wh", "fps"])
    video_info = VideoInfo(resolution_wh, int(fps))
    estimator = SpeedEstimator(video_info, **YOLO_CONFIG)

    while running.value:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_speed_annotation = estimator.process_frame(frame)
            if not annotation_frame_queue.full():
                annotation_frame_queue.put(frame_speed_annotation)

if __name__ == "__main__":
    app = App()
    app.mainloop()

