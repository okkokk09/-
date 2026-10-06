import os
import sys
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# เมื่อรันเป็น .exe แบบไม่มีหน้าต่าง Console (windowed) sys.stdout/stderr จะเป็น None
# ไลบรารีที่พิมพ์ progress (เช่น tqdm ตอนดาวน์โหลดโมเดล) จะ error จึงต้องชี้ไปที่ devnull
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from utils import (
    format_seconds,
    format_file_size,
    get_audio_duration,
    open_folder_in_explorer,
    setup_cuda_dll_paths,
    parse_time_str,
    check_cuda_hardware_status
)
from transcriber import TranscriptionTask

# ตรวจสอบ DLL ของ CUDA ล่วงหน้า
setup_cuda_dll_paths()

# ตั้งค่าธีม CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ฟอนต์ภาษาไทยหลักที่อ่านง่าย คมชัด และสบายตาที่สุดบนระบบ Windows
FONT_FAMILY = "Leelawadee UI"


class MeetingTranscriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ตั้งค่าขนาดหน้าต่างและชื่อโปรแกรม
        self.title("ระบบถอดเสียงการประชุม AI (Speech-to-Text) - GTX 1660 Super Edition")
        self.geometry("1000x800")
        self.minsize(880, 680)

        # ตัวแปรข้อมูล
        self.selected_file_path = None
        self.file_duration = 0.0
        self.current_output_file = None
        self.transcription_task = None
        self.msg_queue = queue.Queue()
        self.cuda_is_ready = False
        self.cuda_detail = ""

        # สร้าง UI
        self._build_ui()

        # ตรวจสอบสถานะ GPU
        self._check_hardware()

        # Loop เช็ค Queue จาก Worker Thread เพื่ออัปเดต UI แบบ Thread-safe
        self.after(100, self._process_queue)

        # ดักจับการปิดหน้าต่าง
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _check_hardware(self):
        """ตรวจสอบและแสดงสถานะ Hardware แบบครอบคลุม"""
        self.cuda_is_ready, badge_text, badge_color, self.cuda_detail = check_cuda_hardware_status()
        self.lbl_hw_status.configure(
            text=badge_text,
            fg_color=badge_color,
            text_color="#34D399" if self.cuda_is_ready else "#FBBF24"
        )
        if self.cuda_is_ready:
            self.cmb_device.set("CUDA (GPU)")
        else:
            self.cmb_device.set("CPU")

    def _show_gpu_diagnostics(self, event=None):
        """แสดงหน้าต่างรายละเอียดสถานะ GPU และวิธีแก้ไข"""
        messagebox.showinfo("รายละเอียดสถานะฮาร์ดแวร์ GPU / CUDA", self.cuda_detail)

    def _build_ui(self):
        # Frame หลัก
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # ---------------- 1. ส่วน Header ----------------
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E222D")
        self.header_frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        title_subframe = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_subframe.pack(side="left", padx=16, pady=12)

        lbl_title = ctk.CTkLabel(
            title_subframe,
            text="🎙️ ระบบถอดเสียงการประชุม AI (Meeting Transcriber)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=20, weight="bold"),
            text_color="#F3F4F6"
        )
        lbl_title.pack(anchor="w")

        lbl_subtitle = ctk.CTkLabel(
            title_subframe,
            text="รองรับไฟล์เสียงความยาวหลายชั่วโมง • ปรับแต่งสำหรับ NVIDIA GTX 1660 Super (VRAM 6GB)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#9CA3AF"
        )
        lbl_subtitle.pack(anchor="w")

        self.lbl_hw_status = ctk.CTkLabel(
            self.header_frame,
            text="  ● กำลังตรวจสอบฮาร์ดแวร์...  ",
            corner_radius=8,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            cursor="hand2"
        )
        self.lbl_hw_status.pack(side="right", padx=16, pady=12)
        self.lbl_hw_status.bind("<Button-1>", self._show_gpu_diagnostics)

        # ---------------- 2. ส่วนเลือกไฟล์และการตั้งค่า ----------------
        self.config_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E222D")
        self.config_frame.grid(row=1, column=0, padx=16, pady=6, sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)

        # แถว 0: เลือกไฟล์
        self.btn_select_file = ctk.CTkButton(
            self.config_frame,
            text="📁 เลือกไฟล์เสียง/วิดีโอ",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            width=180,
            height=38,
            corner_radius=8,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self._choose_file
        )
        self.btn_select_file.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        self.lbl_file_info = ctk.CTkLabel(
            self.config_frame,
            text="ยังไม่ได้เลือกไฟล์ (รองรับ .mp3, .wav, .m4a, .mp4, .aac, .flac)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#9CA3AF",
            anchor="w"
        )
        self.lbl_file_info.grid(row=0, column=1, columnspan=3, padx=8, pady=(14, 6), sticky="ew")

        # แถว 1: ตัวเลือก Model, Device, Language
        options_frame_1 = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        options_frame_1.grid(row=1, column=0, columnspan=4, padx=14, pady=(4, 6), sticky="ew")

        # Model Dropdown
        ctk.CTkLabel(
            options_frame_1,
            text="ขนาดโมเดล:",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            text_color="#D1D5DB"
        ).pack(side="left", padx=(0, 6))

        self.cmb_model = ctk.CTkComboBox(
            options_frame_1,
            values=["tiny", "base", "small", "medium"],
            width=115,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            state="readonly"
        )
        self.cmb_model.set("small")  # Default เป็น small ตามข้อกำหนด
        self.cmb_model.pack(side="left", padx=(0, 20))

        # Device Dropdown
        ctk.CTkLabel(
            options_frame_1,
            text="อุปกรณ์ประมวลผล:",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            text_color="#D1D5DB"
        ).pack(side="left", padx=(0, 6))

        self.cmb_device = ctk.CTkComboBox(
            options_frame_1,
            values=["CUDA (GPU)", "CPU"],
            width=140,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            state="readonly"
        )
        self.cmb_device.set("CUDA (GPU)")  # Default เป็น CUDA ตามข้อกำหนด
        self.cmb_device.pack(side="left", padx=(0, 20))

        # Language Dropdown
        ctk.CTkLabel(
            options_frame_1,
            text="ภาษา:",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            text_color="#D1D5DB"
        ).pack(side="left", padx=(0, 6))

        self.cmb_language = ctk.CTkComboBox(
            options_frame_1,
            values=["ไทย (Thai)", "Auto Detect (อัตโนมัติ)", "อังกฤษ (English)"],
            width=175,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            state="readonly"
        )
        self.cmb_language.set("ไทย (Thai)")
        self.cmb_language.pack(side="left", padx=(0, 10))

        # แถว 2: ตัวเลือกขั้นสูง (Turbo Mode + กำหนดช่วงเวลา)
        options_frame_2 = ctk.CTkFrame(self.config_frame, fg_color="#181B24", corner_radius=8)
        options_frame_2.grid(row=2, column=0, columnspan=4, padx=14, pady=(4, 12), sticky="ew")

        # ⚡ สวิตช์ Turbo Mode
        self.switch_turbo = ctk.CTkSwitch(
            options_frame_2,
            text="⚡ โหมดความเร็วสูง (Turbo Mode: เร็วขึ้น 2-3 เท่า)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            progress_color="#10B981"
        )
        self.switch_turbo.pack(side="left", padx=14, pady=8)

        # เส้นคั่นแนวตั้ง
        sep = ctk.CTkLabel(options_frame_2, text="|", font=ctk.CTkFont(size=14), text_color="#4B5563")
        sep.pack(side="left", padx=8, pady=8)

        # ⏱️ เช็คบ็อกซ์เลือกช่วงเวลา
        self.chk_time_range = ctk.CTkCheckBox(
            options_frame_2,
            text="⏱️ ถอดเฉพาะช่วง:",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            command=self._toggle_time_range
        )
        self.chk_time_range.pack(side="left", padx=(8, 6), pady=8)

        # ช่องกรอกเวลาเริ่มต้น
        self.entry_start_time = ctk.CTkEntry(
            options_frame_2,
            width=85,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            placeholder_text="00:00:00",
            state="disabled"
        )
        self.entry_start_time.pack(side="left", padx=2, pady=8)

        ctk.CTkLabel(
            options_frame_2,
            text="ถึง",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#9CA3AF"
        ).pack(side="left", padx=4, pady=8)

        # ช่องกรอกเวลาสิ้นสุด
        self.entry_end_time = ctk.CTkEntry(
            options_frame_2,
            width=85,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            placeholder_text="00:00:00",
            state="disabled"
        )
        self.entry_end_time.pack(side="left", padx=2, pady=8)

        # ปุ่มรีเซ็ตเป็นทั้งไฟล์
        self.btn_reset_time = ctk.CTkButton(
            options_frame_2,
            text="ทั้งไฟล์",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            width=65,
            height=28,
            fg_color="#374151",
            hover_color="#4B5563",
            state="disabled",
            command=self._reset_time_range
        )
        self.btn_reset_time.pack(side="left", padx=(8, 12), pady=8)

        # ---------------- 3. ส่วนแสดงความคืบหน้า (Progress) ----------------
        self.progress_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E222D")
        self.progress_frame.grid(row=2, column=0, padx=16, pady=6, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)

        # ป้ายสถานะบน
        status_subframe = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        status_subframe.pack(fill="x", padx=14, pady=(10, 4))

        self.lbl_status_text = ctk.CTkLabel(
            status_subframe,
            text="พร้อมเริ่มการถอดเสียง",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            text_color="#38BDF8",
            anchor="w"
        )
        self.lbl_status_text.pack(side="left")

        self.lbl_percentage = ctk.CTkLabel(
            status_subframe,
            text="0%",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color="#F3F4F6",
            anchor="e"
        )
        self.lbl_percentage.pack(side="right")

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=14,
            corner_radius=7,
            progress_color="#3B82F6",
            fg_color="#374151"
        )
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=14, pady=4)

        # ป้ายแสดงเวลา (Elapsed / Total / ETA)
        self.lbl_time_detail = ctk.CTkLabel(
            self.progress_frame,
            text="ตำแหน่งเสียง: 00:00:00 / 00:00:00  •  ใช้เวลาไป: 00:00:00  •  เวลาที่เหลือโดยประมาณ (ETA): --:--:--",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color="#9CA3AF"
        )
        self.lbl_time_detail.pack(anchor="w", padx=14, pady=(2, 10))

        # ---------------- 4. ส่วน Text Box แสดงผลลัพธ์แบบ Real-time ----------------
        self.output_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E222D")
        self.output_frame.grid(row=3, column=0, padx=16, pady=6, sticky="nsew")
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(1, weight=1)

        # Title ของ Text Box พร้อมปุ่ม Copy
        tb_header = ctk.CTkFrame(self.output_frame, fg_color="transparent")
        tb_header.grid(row=0, column=0, padx=14, pady=(10, 4), sticky="ew")

        lbl_output_title = ctk.CTkLabel(
            tb_header,
            text="📝 ข้อความที่ถอดได้ (Real-time Transcript)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            text_color="#E5E7EB"
        )
        lbl_output_title.pack(side="left")

        self.btn_copy_text = ctk.CTkButton(
            tb_header,
            text="📋 คัดลอกข้อความ",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            width=115,
            height=28,
            corner_radius=6,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._copy_to_clipboard
        )
        self.btn_copy_text.pack(side="right")

        self.btn_clear_text = ctk.CTkButton(
            tb_header,
            text="🗑️ ล้างหน้าจอ",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            width=95,
            height=28,
            corner_radius=6,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._clear_textbox
        )
        self.btn_clear_text.pack(side="right", padx=6)

        # กล่อง Text Box รองรับ Auto-scroll ฟอนต์ขนาด 14 อ่านง่าย สบายตา
        self.txt_output = ctk.CTkTextbox(
            self.output_frame,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            corner_radius=8,
            fg_color="#13161C",
            text_color="#F9FAFB",
            wrap="word"
        )
        self.txt_output.grid(row=1, column=0, padx=14, pady=(4, 12), sticky="nsew")

        # ---------------- 5. ส่วนปุ่มควบคุม Action Bar ----------------
        self.action_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E222D")
        self.action_frame.grid(row=4, column=0, padx=16, pady=(6, 16), sticky="ew")

        # ปุ่มเริ่มถอดเสียง
        self.btn_start = ctk.CTkButton(
            self.action_frame,
            text="▶ เริ่มถอดเสียง",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            width=165,
            height=42,
            corner_radius=8,
            fg_color="#10B981",
            hover_color="#059669",
            command=self._start_transcription
        )
        self.btn_start.pack(side="left", padx=14, pady=12)

        # ปุ่มหยุด (Cancel)
        self.btn_stop = ctk.CTkButton(
            self.action_frame,
            text="⏹ หยุด (Cancel)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            width=145,
            height=42,
            corner_radius=8,
            fg_color="#EF4444",
            hover_color="#DC2626",
            state="disabled",
            command=self._stop_transcription
        )
        self.btn_stop.pack(side="left", padx=(0, 14), pady=12)

        # ปุ่มบันทึกเป็น .txt แยกต่างหาก
        self.btn_save_as = ctk.CTkButton(
            self.action_frame,
            text="💾 บันทึกไฟล์เป็น .txt",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            width=155,
            height=42,
            corner_radius=8,
            fg_color="#4B5563",
            hover_color="#6B7280",
            command=self._save_as_txt
        )
        self.btn_save_as.pack(side="right", padx=14, pady=12)

        # ปุ่มเปิดโฟลเดอร์ผลลัพธ์
        self.btn_open_folder = ctk.CTkButton(
            self.action_frame,
            text="📂 เปิดโฟลเดอร์ผลลัพธ์",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            width=165,
            height=42,
            corner_radius=8,
            fg_color="#4B5563",
            hover_color="#6B7280",
            command=self._open_output_folder
        )
        self.btn_open_folder.pack(side="right", padx=(0, 0), pady=12)

    # ---------------- ฟังก์ชันการทำงาน Event Handlers ----------------

    def _toggle_time_range(self):
        """สลับสถานะเปิด/ปิดช่องกรอกเวลาช่วงที่ต้องการถอด"""
        if self.chk_time_range.get() == 1:
            self.entry_start_time.configure(state="normal")
            self.entry_end_time.configure(state="normal")
            self.btn_reset_time.configure(state="normal")
            if not self.entry_start_time.get().strip():
                self.entry_start_time.insert(0, "00:00:00")
            if not self.entry_end_time.get().strip():
                end_str = format_seconds(self.file_duration) if self.file_duration > 0 else "00:00:00"
                self.entry_end_time.insert(0, end_str)
        else:
            self.entry_start_time.configure(state="disabled")
            self.entry_end_time.configure(state="disabled")
            self.btn_reset_time.configure(state="disabled")

    def _reset_time_range(self):
        """รีเซ็ตช่วงเวลากลับเป็นทั้งไฟล์"""
        self.entry_start_time.delete(0, "end")
        self.entry_start_time.insert(0, "00:00:00")
        self.entry_end_time.delete(0, "end")
        if self.file_duration > 0:
            self.entry_end_time.insert(0, format_seconds(self.file_duration))
        else:
            self.entry_end_time.insert(0, "00:00:00")

    def _choose_file(self):
        """เลือกไฟล์เสียงหรือวิดีโอ"""
        filetypes = [
            ("ไฟล์เสียงและวิดีโอที่รองรับ", "*.mp3 *.wav *.m4a *.mp4 *.aac *.flac *.ogg *.wma *.mkv"),
            ("Audio Files (*.mp3, *.wav, *.m4a, *.aac, *.flac)", "*.mp3 *.wav *.m4a *.aac *.flac"),
            ("Video Files (*.mp4, *.mkv)", "*.mp4 *.mkv"),
            ("All Files (*.*)", "*.*")
        ]
        chosen = filedialog.askopenfilename(title="เลือกไฟล์เสียงสำหรับการประชุม", filetypes=filetypes)
        if not chosen:
            return

        self.selected_file_path = chosen
        file_size = os.path.getsize(chosen)
        file_size_str = format_file_size(file_size)
        file_name = os.path.basename(chosen)

        # ดึงระยะเวลา
        self.lbl_file_info.configure(text=f"กำลังตรวจสอบไฟล์ {file_name}...")
        self.update_idletasks()

        duration = get_audio_duration(chosen)
        self.file_duration = duration if duration else 0.0

        if self.file_duration > 0:
            dur_str = format_seconds(self.file_duration)
            self.lbl_file_info.configure(
                text=f"📄 {file_name}  •  ขนาด: {file_size_str}  •  ความยาว: {dur_str}",
                text_color="#6EE7B7"
            )
            self.lbl_time_detail.configure(
                text=f"ตำแหน่งเสียง: 00:00:00 / {dur_str}  •  ใช้เวลาไป: 00:00:00  •  เวลาที่เหลือโดยประมาณ (ETA): --:--:--"
            )
            # เติมเวลาช่วงเริ่มต้นและสิ้นสุดให้อัตโนมัติ
            self.entry_start_time.configure(state="normal")
            self.entry_start_time.delete(0, "end")
            self.entry_start_time.insert(0, "00:00:00")

            self.entry_end_time.configure(state="normal")
            self.entry_end_time.delete(0, "end")
            self.entry_end_time.insert(0, dur_str)

            if self.chk_time_range.get() != 1:
                self.entry_start_time.configure(state="disabled")
                self.entry_end_time.configure(state="disabled")
        else:
            self.lbl_file_info.configure(
                text=f"📄 {file_name}  •  ขนาด: {file_size_str}  •  ความยาว: (ไม่สามารถคำนวณได้)",
                text_color="#FBBF24"
            )

        # กำหนดชื่อไฟล์ output เริ่มต้น
        base_dir = os.path.dirname(chosen)
        base_name = os.path.splitext(file_name)[0]
        self.current_output_file = os.path.join(base_dir, f"{base_name}_transcript.txt")

    def _start_transcription(self):
        """เริ่มกระบวนการถอดเสียง"""
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            messagebox.showwarning("แจ้งเตือน", "กรุณาเลือกไฟล์เสียงหรือวิดีโอก่อนเริ่มถอดเสียง")
            return

        if self.transcription_task and self.transcription_task.is_running():
            return

        # อ่านค่าจาก UI
        model_size = self.cmb_model.get().strip()
        device_choice = "cuda" if "cuda" in self.cmb_device.get().lower() else "cpu"
        
        # ตรวจสอบความพร้อมของ GPU หากผู้ใช้เลือก CUDA
        if device_choice == "cuda" and not self.cuda_is_ready:
            confirm = messagebox.askyesno(
                "การ์ดจอ (GPU) ยังไม่พร้อมใช้งาน",
                f"{self.cuda_detail}\n\n"
                "--------------------------------------------------\n"
                "คุณต้องการสลับไปประมวลผลด้วย 'CPU' ชั่วคราวเพื่อเริ่มถอดเสียงต่อทันทีหรือไม่?"
            )
            if confirm:
                self.cmb_device.set("CPU")
                device_choice = "cpu"
            else:
                return

        lang_choice = self.cmb_language.get()
        if "ไทย" in lang_choice:
            language = "th"
        elif "อังกฤษ" in lang_choice:
            language = "en"
        else:
            language = None  # auto detect

        # ตรวจสอบสถานะ Turbo Mode
        is_turbo = (self.switch_turbo.get() == 1)
        beam_size = 1 if is_turbo else 5

        # ปรับ compute_type ให้เหมาะสมกับ Hardware
        # ในโหมด Turbo: ใช้ int8_float16 บน CUDA ช่วยเร่งความเร็วและประหยัด VRAM สูงสุด
        if device_choice == "cuda":
            compute_type = "int8_float16" if is_turbo else "float16"
        else:
            compute_type = "int8"

        # ตรวจสอบช่วงเวลาที่เลือก (Custom Time Range)
        clip_timestamps = None
        if self.chk_time_range.get() == 1:
            start_str = self.entry_start_time.get().strip()
            end_str = self.entry_end_time.get().strip()

            start_sec = parse_time_str(start_str)
            end_sec = parse_time_str(end_str)

            if start_sec is None or end_sec is None:
                messagebox.showwarning(
                    "รูปแบบเวลาไม่ถูกต้อง",
                    "กรุณาระบุช่วงเวลาในรูปแบบ HH:MM:SS เช่น 00:15:00 หรือ 01:30:00"
                )
                return

            if start_sec >= end_sec:
                messagebox.showwarning(
                    "ช่วงเวลาไม่ถูกต้อง",
                    "เวลาเริ่มต้นต้องน้อยกว่าเวลาสิ้นสุดครับ"
                )
                return

            if self.file_duration > 0 and start_sec > self.file_duration:
                messagebox.showwarning(
                    "ช่วงเวลาไม่ถูกต้อง",
                    f"เวลาเริ่มต้น ({format_seconds(start_sec)}) เกินความยาวทั้งหมดของไฟล์ ({format_seconds(self.file_duration)})"
                )
                return

            clip_timestamps = [start_sec, end_sec]

        # ตั้งค่าสถานะ UI ขณะทำงาน
        self.btn_start.configure(state="disabled")
        self.btn_select_file.configure(state="disabled")
        self.cmb_model.configure(state="disabled")
        self.cmb_device.configure(state="disabled")
        self.cmb_language.configure(state="disabled")
        self.switch_turbo.configure(state="disabled")
        self.chk_time_range.configure(state="disabled")
        self.entry_start_time.configure(state="disabled")
        self.entry_end_time.configure(state="disabled")
        self.btn_reset_time.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self.progress_bar.set(0.0)
        self.lbl_percentage.configure(text="0%")
        turbo_tag = " [โหมด Turbo ⚡]" if is_turbo else ""
        self.lbl_status_text.configure(text=f"กำลังเตรียมเริ่มการถอดเสียง{turbo_tag}...", text_color="#38BDF8")

        # เริ่ม Background Task
        self.transcription_task = TranscriptionTask(
            audio_path=self.selected_file_path,
            model_size=model_size,
            device=device_choice,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            clip_timestamps=clip_timestamps,
            is_turbo=is_turbo,
            output_txt_path=self.current_output_file,
            on_status=lambda s: self.msg_queue.put(("status", s)),
            on_progress=lambda p, c, t, el, eta: self.msg_queue.put(("progress", (p, c, t, el, eta))),
            on_segment=lambda ts, txt, line: self.msg_queue.put(("segment", (ts, txt, line))),
            on_finished=lambda success, msg, path: self.msg_queue.put(("finished", (success, msg, path))),
            on_error=lambda err: self.msg_queue.put(("error", err))
        )
        self.transcription_task.start()

    def _stop_transcription(self):
        """ยกเลิกการถอดเสียง"""
        if self.transcription_task and self.transcription_task.is_running():
            self.lbl_status_text.configure(text="กำลังหยุดการทำงาน...", text_color="#F87171")
            self.transcription_task.cancel()
            self.btn_stop.configure(state="disabled")

    def _process_queue(self):
        """ดึงข้อมูลจาก Queue มาอัปเดต UI ใน Main Thread"""
        try:
            while not self.msg_queue.empty():
                msg_type, data = self.msg_queue.get_nowait()
                
                if msg_type == "status":
                    self.lbl_status_text.configure(text=data)

                elif msg_type == "progress":
                    pct, cur, tot, el, eta = data
                    self.progress_bar.set(pct / 100.0)
                    self.lbl_percentage.configure(text=f"{pct:.1f}%")
                    self.lbl_time_detail.configure(
                        text=f"ตำแหน่งเสียง: {format_seconds(cur)} / {format_seconds(tot)}  •  "
                             f"ใช้เวลาไป: {format_seconds(el)}  •  "
                             f"เวลาที่เหลือโดยประมาณ (ETA): {format_seconds(eta)}"
                    )

                elif msg_type == "segment":
                    ts_str, text_str, line_str = data
                    # ใส่ลงใน Text Box และ Auto-scroll
                    self.txt_output.insert("end", line_str)
                    self.txt_output.see("end")

                elif msg_type == "finished":
                    success, msg, path = data
                    self._on_task_completed(success, msg, path)

                elif msg_type == "error":
                    self._on_task_error(data)

        except Exception:
            pass

        # วนรอบตรวจสอบ Queue ต่อไปทุก 100ms
        self.after(100, self._process_queue)

    def _on_task_completed(self, success: bool, message: str, output_path: str):
        """จัดการเมื่อ Task จบการทำงาน (สำเร็จหรือยกเลิก)"""
        self.btn_start.configure(state="normal")
        self.btn_select_file.configure(state="normal")
        self.cmb_model.configure(state="normal")
        self.cmb_device.configure(state="normal")
        self.cmb_language.configure(state="normal")
        self.switch_turbo.configure(state="normal")
        self.chk_time_range.configure(state="normal")
        self._toggle_time_range()
        self.btn_stop.configure(state="disabled")

        if success:
            self.lbl_status_text.configure(text="🎉 การถอดเสียงเสร็จสมบูรณ์!", text_color="#34D399")
            self.lbl_percentage.configure(text="100%")
            messagebox.showinfo(
                "สำเร็จ",
                f"การถอดเสียงเสร็จสมบูรณ์เรียบร้อยแล้ว!\n\n{message}\n\nบันทึกไฟล์ไว้ที่:\n{output_path}"
            )
        else:
            self.lbl_status_text.configure(text="⏹ หยุดการทำงานแล้ว", text_color="#FBBF24")

    def _on_task_error(self, err_msg: str):
        """จัดการเมื่อเกิด Error"""
        self.btn_start.configure(state="normal")
        self.btn_select_file.configure(state="normal")
        self.cmb_model.configure(state="normal")
        self.cmb_device.configure(state="normal")
        self.cmb_language.configure(state="normal")
        self.switch_turbo.configure(state="normal")
        self.chk_time_range.configure(state="normal")
        self._toggle_time_range()
        self.btn_stop.configure(state="disabled")

        self.lbl_status_text.configure(text="❌ เกิดข้อผิดพลาด", text_color="#EF4444")

        # ตรวจจับข้อผิดพลาดเรื่อง CUDA เพื่อเสนอทางเลือกสลับไป CPU อัตโนมัติ
        err_lower = err_msg.lower()
        if any(k in err_lower for k in ["cuda", "cublas", "cudnn", "driver", "insufficient"]):
            confirm = messagebox.askyesno(
                "เกิดข้อผิดพลาดในการประมวลผลด้วย GPU (CUDA)",
                f"ไม่สามารถประมวลผลบนการ์ดจอได้เนื่องจาก:\n{err_msg}\n\n"
                "--------------------------------------------------\n"
                "💡 คุณต้องการให้ระบบสลับไปใช้ 'CPU' อัตโนมัติเพื่อเริ่มถอดเสียงต่อทันทีหรือไม่?"
            )
            if confirm:
                self.cmb_device.set("CPU")
                self._start_transcription()
                return

        messagebox.showerror("เกิดข้อผิดพลาดในการประมวลผล", err_msg)

    def _copy_to_clipboard(self):
        """คัดลอกข้อความใน Text Box ทั้งหมดไปยัง Clipboard"""
        content = self.txt_output.get("1.0", "end").strip()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            self.update()
            messagebox.showinfo("คัดลอกสำเร็จ", "คัดลอกข้อความทั้งหมดไปยังคลิปบอร์ดเรียบร้อยแล้ว")
        else:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อความสำหรับคัดลอก")

    def _clear_textbox(self):
        """ล้างหน้าจอข้อความ"""
        if messagebox.askyesno("ยืนยัน", "ต้องการล้างข้อความบนหน้าจอหรือไม่? (ไฟล์ที่บันทึกไว้จะไม่หาย)"):
            self.txt_output.delete("1.0", "end")

    def _open_output_folder(self):
        """เปิดโฟลเดอร์ผลลัพธ์ใน File Explorer"""
        if self.current_output_file and os.path.exists(os.path.dirname(self.current_output_file)):
            target_dir = os.path.dirname(self.current_output_file)
        elif self.selected_file_path and os.path.exists(os.path.dirname(self.selected_file_path)):
            target_dir = os.path.dirname(self.selected_file_path)
        else:
            target_dir = os.getcwd()

        open_folder_in_explorer(target_dir)

    def _save_as_txt(self):
        """บันทึกข้อความปัจจุบันลงไฟล์ .txt ตามที่ผู้ใช้กำหนด"""
        content = self.txt_output.get("1.0", "end").strip()
        if not content:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อความสำหรับบันทึก")
            return

        default_name = "meeting_transcript.txt"
        if self.selected_file_path:
            base_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
            default_name = f"{base_name}_transcript.txt"

        save_path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์เป็น .txt",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text File (*.txt)", "*.txt"), ("All Files (*.*)", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("บันทึกสำเร็จ", f"บันทึกไฟล์เรียบร้อยแล้วที่:\n{save_path}")
            except Exception as e:
                messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถบันทึกไฟล์ได้:\n{str(e)}")

    def _on_close(self):
        """ดักจับตอนปิดโปรแกรม"""
        if self.transcription_task and self.transcription_task.is_running():
            if messagebox.askyesno("ยืนยันการปิดโปรแกรม", "การถอดเสียงกำลังทำงานอยู่ คุณต้องการหยุดและปิดโปรแกรมหรือไม่?"):
                self.transcription_task.cancel()
                self.destroy()
        else:
            self.destroy()


if __name__ == "__main__":
    app = MeetingTranscriberApp()
    app.mainloop()
