import os
import sys
import time
import threading
from typing import Callable, Optional, List
from utils import format_seconds, get_audio_duration, setup_cuda_dll_paths

# โหลดพาธ DLL ของ CUDA ล่วงหน้า
setup_cuda_dll_paths()


class TranscriptionTask:
    """
    คลาสจัดการการถอดเสียงด้วย faster-whisper ใน Background Worker Thread
    รองรับ:
    - การถอดเสียงแบบ Real-time พร้อม Auto-scroll และ Append ลงไฟล์
    - การควบคุมผ่าน Cancel Event
    - โหมดความเร็วสูง (Turbo Mode: beam_size=1, int8_float16)
    - การกำหนดช่วงเวลาที่ต้องการถอดเสียง (clip_timestamps: [start_sec, end_sec])
    """
    def __init__(
        self,
        audio_path: str,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: int = 5,
        clip_timestamps: Optional[List[float]] = None,
        is_turbo: bool = False,
        output_txt_path: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float, float, float, float, float], None]] = None,
        on_segment: Optional[Callable[[str, str, str], None]] = None,
        on_finished: Optional[Callable[[bool, str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.audio_path = audio_path
        self.model_size = model_size
        self.device = device.lower()
        self.language = language if language and language != "auto" else None
        self.beam_size = beam_size
        self.clip_timestamps = clip_timestamps
        self.is_turbo = is_turbo
        
        # ปรับ compute_type ให้เหมาะสมกับ Hardware
        # GTX 1660 Super (Turing) รองรับ int8_float16 และ float16 ได้อย่างเสถียร
        if compute_type:
            self.compute_type = compute_type
        else:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        # หากไม่ได้ระบุ output path ให้สร้าง transcript.txt ในโฟลเดอร์เดียวกับไฟล์เสียง
        if output_txt_path:
            self.output_txt_path = output_txt_path
        else:
            base_dir = os.path.dirname(self.audio_path)
            base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
            self.output_txt_path = os.path.join(base_dir, f"{base_name}_transcript.txt")

        # Callbacks
        self.on_status = on_status or (lambda s: None)
        self.on_progress = on_progress or (lambda p, c, t, el, eta: None)
        self.on_segment = on_segment or (lambda ts, txt, line: None)
        self.on_finished = on_finished or (lambda s, m, p: None)
        self.on_error = on_error or (lambda e: None)

        self.cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """เริ่มการทำงานใน Worker Thread แยก เพื่อไม่ให้ GUI ค้าง"""
        self.cancel_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        """ส่งสัญญาณยกเลิกการถอดเสียง"""
        self.cancel_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        start_wall_time = time.time()
        try:
            self.on_status("กำลังตรวจสอบความยาวไฟล์เสียง...")
            total_duration = get_audio_duration(self.audio_path)
            if total_duration is None or total_duration <= 0:
                total_duration = 1.0

            turbo_info = " [โหมด Turbo ⚡]" if self.is_turbo else ""
            self.on_status(f"กำลังโหลดโมเดล AI '{self.model_size}' บน {self.device.upper()} ({self.compute_type}){turbo_info}...")
            
            # โหลด faster-whisper
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "ไม่พบไลบรารี faster-whisper กรุณาติดตั้งด้วยคำสั่ง:\n"
                    "pip install faster-whisper"
                )

            # พยายามโหลดโมเดลพร้อมระบบ Fallback หาก compute_type ไม่รองรับ
            try:
                model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None
                )
            except ValueError as ve:
                ve_str = str(ve).lower()
                if "compute_type" in ve_str or "unsupported" in ve_str:
                    fallback_type = "float16" if self.device == "cuda" else "int8"
                    self.compute_type = fallback_type
                    model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=fallback_type,
                        download_root=None
                    )
                else:
                    raise ve
            except Exception as e:
                err_str = str(e)
                # ดักจับกรณี CUDA / cuDNN DLL ไม่สมบูรณ์บน Windows
                if self.device == "cuda" and ("cudnn" in err_str.lower() or "cublas" in err_str.lower() or "cuda" in err_str.lower()):
                    raise RuntimeError(
                        f"เกิดข้อผิดพลาดในการโหลดโมเดลบน GPU CUDA:\n{err_str}\n\n"
                        "คำแนะนำ:\n"
                        "1. ตรวจสอบว่าได้ติดตั้ง Driver NVIDIA และ CUDA/cuDNN เรียบร้อยแล้ว\n"
                        "2. สามารถลองเปลี่ยน Device เป็น 'CPU' ในหน้าต่างโปรแกรมเพื่อประมวลผลได้ทันที"
                    )
                raise e

            if self.cancel_event.is_set():
                self.on_finished(False, "ยกเลิกการทำงานแล้ว", self.output_txt_path)
                return

            self.on_status(f"กำลังเริ่มถอดเสียงการประชุม{turbo_info}...")

            # เตรียมไฟล์ output สำหรับ Real-time append
            output_dir = os.path.dirname(self.output_txt_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # ตรวจสอบช่วงเวลาที่เลือก (Clip Timestamps)
            if self.clip_timestamps and len(self.clip_timestamps) >= 2:
                clip_start = float(self.clip_timestamps[0])
                clip_end = float(self.clip_timestamps[1])
                effective_duration = max(1.0, clip_end - clip_start)
                range_info_str = f"ช่วงเวลาที่เลือก: {format_seconds(clip_start)} ถึง {format_seconds(clip_end)} (รวม {format_seconds(effective_duration)})"
            else:
                clip_start = 0.0
                clip_end = total_duration
                effective_duration = total_duration
                range_info_str = f"ทั้งไฟล์ (ความยาว: {format_seconds(total_duration)})"

            # เขียน Header ลงไฟล์ผลลัพธ์
            with open(self.output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== รายงานถอดเสียงการประชุม (Speech-to-Text) ===\n")
                f.write(f"ไฟล์ต้นฉบับ: {os.path.basename(self.audio_path)}\n")
                f.write(f"โมเดล: {self.model_size} | อุปกรณ์: {self.device.upper()} ({self.compute_type})\n")
                mode_str = f"ความเร็วสูง (Turbo Mode: beam_size={self.beam_size})" if self.is_turbo else f"มาตรฐาน (beam_size={self.beam_size})"
                f.write(f"โหมดการทำงาน: {mode_str}\n")
                f.write(f"ช่วงเวลาที่ถอดเสียง: {range_info_str}\n")
                f.write(f"วันที่ถอดเสียง: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.flush()

            # กำหนด Parameter สำหรับ transcribe
            transcribe_kwargs = dict(
                language=self.language,
                beam_size=self.beam_size,
            )

            if self.clip_timestamps:
                transcribe_kwargs["clip_timestamps"] = self.clip_timestamps
            else:
                # vad_filter=True ช่วยตัดเสียงเงียบ ลดการประมวลผลที่เปล่าประโยชน์
                transcribe_kwargs["vad_filter"] = True
                transcribe_kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

            segments_gen, info = model.transcribe(self.audio_path, **transcribe_kwargs)

            # อัปเดต total_duration จาก info หากทั้งไฟล์
            if not self.clip_timestamps and info and hasattr(info, "duration") and info.duration:
                total_duration = info.duration
                effective_duration = total_duration
                clip_end = total_duration

            segment_count = 0
            with open(self.output_txt_path, "a", encoding="utf-8") as f:
                for segment in segments_gen:
                    if self.cancel_event.is_set():
                        f.write(f"\n[การถอดเสียงถูกหยุดโดยผู้ใช้ ณ เวลา {format_seconds(segment.end)}]\n")
                        f.flush()
                        self.on_finished(False, "หยุดการทำงานเรียบร้อยแล้ว", self.output_txt_path)
                        return

                    segment_count += 1
                    current_sec = segment.end
                    ts_str = f"[{format_seconds(segment.start)}]"
                    text_str = segment.text.strip()
                    line_str = f"{ts_str} {text_str}\n"

                    # บันทึกลงไฟล์ทันที (Real-time append & flush)
                    f.write(line_str)
                    f.flush()

                    # ส่งข้อมูลกลับ GUI
                    self.on_segment(ts_str, text_str, line_str)

                    # คำนวณความคืบหน้าและเวลา
                    elapsed_sec = time.time() - start_wall_time
                    processed_in_range = max(0.0, current_sec - clip_start)
                    progress_pct = min(100.0, (processed_in_range / effective_duration) * 100.0)
                    
                    if processed_in_range > 0 and progress_pct > 0:
                        speed = processed_in_range / elapsed_sec if elapsed_sec > 0 else 1.0
                        remaining_audio_sec = max(0.0, clip_end - current_sec)
                        eta_sec = remaining_audio_sec / speed if speed > 0 else 0.0
                    else:
                        eta_sec = 0.0

                    self.on_progress(progress_pct, current_sec, clip_end, elapsed_sec, eta_sec)
                    self.on_status(f"กำลังประมวลผล{turbo_info}... (ช่วงเสียง: {format_seconds(current_sec)} / {format_seconds(clip_end)})")

            # เสร็จสิ้นกระบวนการ
            total_elapsed = time.time() - start_wall_time
            self.on_progress(100.0, clip_end, clip_end, total_elapsed, 0.0)
            self.on_status("การถอดเสียงเสร็จสมบูรณ์!")
            self.on_finished(
                True,
                f"ถอดเสียงเสร็จสิ้น {segment_count} ประโยค ({range_info_str})\nใช้เวลาประมวลผลทั้งหมด: {format_seconds(total_elapsed)}",
                self.output_txt_path
            )

        except Exception as e:
            err_msg = str(e)
            self.on_error(err_msg)
            self.on_finished(False, f"เกิดข้อผิดพลาด: {err_msg}", self.output_txt_path)
