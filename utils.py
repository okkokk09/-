import os
import sys
import subprocess
import shutil
from typing import Optional, Tuple


def _patch_av_open_compat() -> None:
    """
    PyAV >= 14 ยกเลิกพารามิเตอร์ metadata_errors ใน av.open()
    แต่ faster-whisper (audio.py) ยังส่งพารามิเตอร์นี้อยู่ ทำให้เกิด
    TypeError: open() got an unexpected keyword argument 'metadata_errors'
    ฟังก์ชันนี้ครอบ av.open ให้ตัดพารามิเตอร์ที่ไม่รองรับทิ้งอัตโนมัติ
    """
    try:
        import av
    except Exception:
        return
    if getattr(av.open, "_metadata_errors_patched", False):
        return
    _orig_open = av.open

    def _open(*args, **kwargs):
        try:
            return _orig_open(*args, **kwargs)
        except TypeError as e:
            if "metadata_errors" in kwargs and "metadata_errors" in str(e):
                kwargs.pop("metadata_errors", None)
                return _orig_open(*args, **kwargs)
            raise

    _open._metadata_errors_patched = True
    av.open = _open


_patch_av_open_compat()

def setup_cuda_dll_paths() -> bool:
    """
    ตรวจจับและเพิ่มพาธ DLL ของ NVIDIA (cuBLAS, cuDNN) บน Windows 
    และคัดลอก DLL ที่จำเป็นเข้าโฟลเดอร์ ctranslate2 โดยตรงเพื่อให้ CTranslate2 เรียกใช้ได้ 100%
    """
    if sys.platform != "win32":
        return True

    found_any = False
    dll_dirs = []

    # 1. ตรวจสอบใน site-packages
    try:
        import site
        site_packages = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_packages.append(site.getusersitepackages())

        ct2_dir = None
        try:
            import ctranslate2
            ct2_dir = os.path.dirname(ctranslate2.__file__)
        except Exception:
            pass

        for sp in site_packages:
            nvidia_dir = os.path.join(sp, "nvidia")
            if os.path.exists(nvidia_dir):
                for root, dirs, files in os.walk(nvidia_dir):
                    dll_files = [f for f in files if f.lower().endswith(".dll")]
                    if dll_files:
                        dll_dirs.append(root)
                        found_any = True
                        # คัดลอก DLL ลงใน ctranslate2 dir โดยตรงถ้ายังไม่มี
                        if ct2_dir and os.path.exists(ct2_dir):
                            for f in dll_files:
                                dst = os.path.join(ct2_dir, f)
                                if not os.path.exists(dst):
                                    try:
                                        shutil.copy2(os.path.join(root, f), dst)
                                    except Exception:
                                        pass
    except Exception:
        pass

    # 2. ตรวจสอบ PATH ปกติของ CUDA Toolkit
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and os.path.exists(os.path.join(cuda_path, "bin")):
        dll_dirs.append(os.path.join(cuda_path, "bin"))
        found_any = True

    # 3. เพิ่มลงทั้ง os.add_dll_directory และ os.environ["PATH"]
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    return found_any


def check_cuda_hardware_status() -> Tuple[bool, str, str, str]:
    """
    ตรวจสอบความพร้อมของ GPU (CUDA) แบบเร็วโดยไม่ต้องโหลดโมเดลใหญ่
    คืนค่า: (is_ready, badge_text, badge_color, detailed_explanation)
    """
    setup_cuda_dll_paths()
    try:
        import ctranslate2
        # ทดสอบเรียก compute types
        compute_types = ctranslate2.get_supported_compute_types("cuda")
        device_count = ctranslate2.get_cuda_device_count()
        if device_count > 0 and compute_types:
            return (
                True,
                "  ● GPU พร้อมใช้งาน (CUDA)  ",
                "#064E3B",  # เขียว
                f"ตรวจพบ GPU NVIDIA {device_count} อุปกรณ์ พร้อมใช้งานด้วย CUDA\nรองรับโหมดความเร็ว: {', '.join(compute_types)}"
            )
        else:
            return (
                False,
                "  ● โหมด CPU (ไม่พบการ์ดจอ NVIDIA)  ",
                "#374151",  # เทา
                "ไม่พบการ์ดจอ NVIDIA ที่รองรับ CUDA บนเครื่องนี้\nโปรแกรมจะทำงานในโหมด 'CPU' อัตโนมัติ"
            )
    except Exception as e:
        err_msg = str(e)
        if "driver version is insufficient" in err_msg.lower():
            return (
                False,
                "  ⚠️ Driver การ์ดจอเก่าเกินไป (คลิกดูวิธีแก้)  ",
                "#78350F",  # ส้มอำพัน
                "ตรวจพบการ์ดจอ NVIDIA แต่ไดรเวอร์ (Driver) ของการ์ดจอบนเครื่องนี้เป็นเวอร์ชันเก่าเกินไปสำหรับ CUDA 12\n\n"
                "📌 วิธีแก้ไขเพื่อให้ใช้ GPU ได้:\n"
                "1. ไปที่เว็บไซต์ https://www.nvidia.com/Download\n"
                "2. เลือกรุ่นการ์ดจอของคุณ (GeForce GTX 1660 Super) แล้วดาวน์โหลด Driver เวอร์ชันล่าสุดมาติดตั้ง\n"
                "3. หรืออัปเดตผ่านโปรแกรม NVIDIA GeForce Experience\n\n"
                "💡 ขณะนี้คุณยังสามารถใช้งานโปรแกรมได้ตามปกติโดยเลือกอุปกรณ์เป็น 'CPU'"
            )
        elif "cublas" in err_msg.lower() or "cudnn" in err_msg.lower():
            return (
                False,
                "  ⚠️ ไม่พบไฟล์ CUDA DLL (คลิกดูวิธีแก้)  ",
                "#78350F",
                "ไม่พบไฟล์ไลบรารี DLL ของ CUDA/cuDNN สำหรับ Windows\n\n"
                "📌 วิธีแก้ไข:\n"
                "กรุณาดับเบิลคลิกเปิดไฟล์ 'install.bat' ใหม่อีกครั้งเพื่อติดตั้งไลบรารีให้ครบถ้วน\n\n"
                "💡 หรือสามารถเลือกอุปกรณ์เป็น 'CPU' เพื่อใช้งานได้ทันที"
            )
        else:
            return (
                False,
                "  ● GPU ไม่พร้อม (คลิกดูรายละเอียด)  ",
                "#78350F",
                f"GPU ยังไม่พร้อมใช้งาน:\n{err_msg}\n\nแนะนำให้เลือกอุปกรณ์เป็น 'CPU' เพื่อถอดเสียง"
            )



def format_seconds(seconds: float) -> str:
    """
    แปลงจำนวนวินาทีเป็นสตริงรูปแบบ HH:MM:SS
    เช่น 3661 -> '01:01:01'
    """
    if seconds < 0:
        seconds = 0
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_file_size(size_in_bytes: int) -> str:
    """
    แปลงขนาดไบต์เป็นหน่วยที่อ่านง่าย (B, KB, MB, GB)
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    คำนวณความยาวไฟล์เสียง (วินาที)
    พยายามใช้ PyAV ก่อน (faster-whisper มี av อยู่แล้ว) หากไม่มีใช้ ffprobe
    """
    if not os.path.exists(file_path):
        return None

    # พยายามใช้ PyAV
    try:
        import av
        with av.open(file_path) as container:
            if container.duration is not None:
                return float(container.duration) / av.time_base
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass

    # พยายามใช้ ffprobe ถ้ามี
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


def open_folder_in_explorer(folder_path: str):
    """
    เปิดโฟลเดอร์ใน Windows Explorer หรือ File Manager ของระบบ
    """
    abs_path = os.path.abspath(folder_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path, exist_ok=True)

    if sys.platform == "win32":
        os.startfile(abs_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", abs_path])
    else:
        subprocess.Popen(["xdg-open", abs_path])


def parse_time_str(time_str: str) -> Optional[float]:
    """
    แปลงข้อความเวลา (เช่น '01:30:00', '15:30', '90') เป็นวินาที (float)
    คืนค่า None หากรูปแบบไม่ถูกต้อง
    """
    if not time_str or not time_str.strip():
        return None
    s = time_str.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            if h < 0 or m < 0 or m >= 60 or sec < 0 or sec >= 60:
                return None
            return h * 3600 + m * 60 + sec
        elif len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            if m < 0 or sec < 0 or sec >= 60:
                return None
            return m * 60 + sec
        elif len(parts) == 1:
            val = float(parts[0])
            return val if val >= 0 else None
    except ValueError:
        return None
    return None

