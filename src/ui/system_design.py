from dataclasses import dataclass

@dataclass
class ColorStyle:
    hover_brand_dark_color: str = "#005A5A"
    hover_brand_light_color: str = "#E4F9F9"
    hover_gray_light_color: str = "#F8FAFC"
    hover_gray_dark_color: str = "#E2E8F0"
    
    text_brand_color: str = "#00A2A2"
    text_black_color: str = "#334155"
    text_white_color: str = "#FFFFFF"
    text_gray_light_color: str = "#94A3B8"
    text_gray_dark_color: str = "#475569"
    
    bg_brand_color: str = "#00A2A2"
    bg_brand_dark_color: str = "#CCECEC"
    bg_brand_light_color: str = "#E4F9F9"
    bg_white_color: str = "#FFFFFF"
    bg_gray_light_color: str = "#F1F5F9"
    
    border_brand_color: str = "#00A2A2"
    border_gray_light_color: str = "#E2E8F0"
    
    icon_black_color: str = "#334155"

 # 🔹 สร้าง Instance ค่าคงที่
COLOR_STYLE = ColorStyle()

@dataclass
class ColorStyle_BGR:
    border_brand_color: str = (162, 162, 0)
 # 🔹 สร้าง Instance ค่าคงที่
COLOR_STYLE_BGR = ColorStyle_BGR()

@dataclass
class ColorStyle_RGB:
    border_brand_color: str = (0, 162, 162)
    border_white_color: str = (255, 255, 255)
 # 🔹 สร้าง Instance ค่าคงที่
COLOR_STYLE_RGB = ColorStyle_RGB()
    
# 🔹 กำหนดค่าคงที่สำหรับ System Design
SYSTEM_DESIGN = {
    "Button-Primary": {
        "fg_color": COLOR_STYLE.bg_brand_color,
        "hover_color": COLOR_STYLE.hover_brand_dark_color,
        "text_color": COLOR_STYLE.text_white_color,
        "border_width": 2,
        "border_color": COLOR_STYLE.border_brand_color,
        "state": "normal"
    },
    "Button-Secondary": {
        "fg_color": COLOR_STYLE.bg_white_color,
        "hover_color": COLOR_STYLE.hover_brand_light_color,
        "text_color": COLOR_STYLE.text_brand_color,
        "border_width": 2,
        "border_color": COLOR_STYLE.border_brand_color,
        "state": "normal"
    },
    "Button-Disabled": {
        "fg_color": COLOR_STYLE.bg_gray_light_color,
        "text_color": COLOR_STYLE.text_gray_light_color,
        "border_width": 2,
        "border_color": COLOR_STYLE.border_gray_light_color,
        "state": "disabled"
    },
    "Navigation": {
        "fg_color": "transparent",
        "text_color": COLOR_STYLE.text_black_color,
        "hover_color": COLOR_STYLE.hover_gray_dark_color
    },
    "Navigation-Clicked": {
        "fg_color": COLOR_STYLE.bg_brand_dark_color,
        "text_color": COLOR_STYLE.text_black_color,
        "hover_color": COLOR_STYLE.hover_gray_dark_color
    },
    "Dropdown": {
        "fg_color": COLOR_STYLE.bg_white_color,  # สีพื้นหลังของ Dropdown
        "border_color": COLOR_STYLE.border_brand_color,  # สีขอบของ Dropdown
        "border_width": 2,  # ความหนาของขอบของ Dropdown
        "button_color": COLOR_STYLE.bg_brand_color,  # สีปุ่ม Dropdown
        "button_hover_color": COLOR_STYLE.hover_brand_dark_color,  # สีปุ่มเมื่อโฮเวอร์
        "dropdown_fg_color": COLOR_STYLE.bg_white_color,  # สีพื้นหลังของ Dropdown
        "dropdown_hover_color": COLOR_STYLE.hover_gray_light_color,  # สีเมื่อโฮเวอร์
        "text_color": COLOR_STYLE.text_black_color  # สีข้อความ
    },
    "Frame-Info": {
        "fg_color": COLOR_STYLE.bg_white_color,
        "border_color": COLOR_STYLE.border_gray_light_color,
        "border_width": 2
    },
    "Input-Field": {
        "fg_color": COLOR_STYLE.bg_white_color,
        "border_color": COLOR_STYLE.border_gray_light_color,
        "border_width": 1,
        "text_color": COLOR_STYLE.text_black_color,
        "placeholder_text_color": COLOR_STYLE.text_gray_light_color,
        "state": "normal"
    },
    "Input-Field-Disabled": {
        "fg_color": COLOR_STYLE.bg_gray_light_color,
        "border_color": COLOR_STYLE.border_gray_light_color,
        "border_width": 1,
        "text_color": COLOR_STYLE.text_gray_light_color,
        "state": "disabled"
    },
    "Label": {
        "text_color": COLOR_STYLE.text_black_color
    },
    "Label-Bold": {
        "text_color": COLOR_STYLE.text_black_color,
        "font": ("Arial", 18)
    }
}

# Eample
# self.start_button = ctk.CTkButton(
#     self.button_frame, text="Start", command=self.start_stream,
#     **SYSTEM_DESIGN["button"]  # นำค่าทั้งหมดมาใช้
# )