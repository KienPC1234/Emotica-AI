# -*- coding: utf-8 -*-
import json
import os
import sys
import signal
import time
import threading
import subprocess
from pathlib import Path
import webbrowser
import toml
from typing import Tuple, Dict, Any, Optional, List

# --- Cấu hình hằng số ---
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs'
LOG_FILE = LOG_DIR / 'latest.log'
CONFIG_FILE = BASE_DIR / 'multiprocess.json'
PORTS_FILE = BASE_DIR / 'ports.toml'
REQUIREMENTS_FILE = BASE_DIR / 'requirements.txt'

MIN_PYTHON_VERSION = (3, 10)
MAX_PYTHON_VERSION = (3, 13)  # < 3.13 (tức là 3.10, 3.11, 3.12)


def rgb_gradient_text(text: str, start_rgb: Tuple[int, int, int] = (255, 0, 0), end_rgb: Tuple[int, int, int] = (0, 0, 255)) -> str:
    """Tạo gradient RGB thật mượt cho chuỗi text."""
    result = ""
    r1, g1, b1 = start_rgb
    r2, g2, b2 = end_rgb
    
    char_count = len(text)
    # Guard against division by zero for single-character strings
    divisor = max(char_count - 1, 1)

    for i, ch in enumerate(text):
        ratio = i / divisor
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        result += f"\033[38;2;{r};{g};{b}m{ch}\033[0m"
    return result


def print_banner():
    """In banner ASCII art với hiệu ứng màu sắc."""
    banner = r"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐  ███╗   ███╗ █████╗ ██╗  ██╗███████╗    ██████╗ ██╗   ██╗                   ▌
▐  ████╗ ████║██╔══██╗██║ ██╔╝██╔════╝    ██╔══██╗╚██╗ ██╔╝                   ▌
▐  ██╔████╔██║███████║█████╔╝ █████╗      ██████╔╝ ╚████╔╝                    ▌
▐  ██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝      ██╔══██╗  ╚██╔╝                     ▌
▐  ██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗    ██████╔╝   ██║                      ▌
▐  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝    ╚═╝                      ▌
▐                                                                             ▌
▐  ██╗  ██╗██╗███████╗███╗   ██╗██████╗  ██████╗ ██╗██████╗ ██████╗ ██╗  ██╗  ▌
▐  ██║ ██╔╝██║██╔════╝████╗  ██║██╔══██╗██╔════╝███║╚════██╗╚════██╗██║  ██║  ▌
▐  █████╔╝ ██║█████╗  ██╔██╗ ██║██████╔╝██║     ╚██║ █████╔╝ █████╔╝███████║  ▌
▐  ██╔═██╗ ██║██╔══╝  ██║╚██╗██║██╔═══╝ ██║      ██║██╔═══╝  ╚═══██╗╚════██║  ▌
▐  ██║  ██╗██║███████╗██║ ╚████║██║     ╚██████╗ ██║███████╗██████╔╝     ██║  ▌
▐  ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚═╝╚══════╝╚═════╝      ╚═╝  ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
Credit: https://github.com/KienPC1234
Repo: https://github.com/KienPC1234/Emotica-AI
"""
    lines = banner.splitlines()
    colors = [
        (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
        (0, 255, 255), (0, 127, 255), (127, 0, 255), (255, 0, 255),
    ]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        print(rgb_gradient_text(line, start_rgb=color, end_rgb=colors[(i + 1) % len(colors)]))
        # Xóa 'time.sleep' nếu bạn muốn banner xuất hiện ngay lập tức
        time.sleep(0.01)
    print("\n")


def check_python_version():
    """Kiểm tra phiên bản Python có nằm trong khoảng yêu cầu không."""
    current_version = sys.version_info
    if not (MIN_PYTHON_VERSION <= current_version < MAX_PYTHON_VERSION):
        print(f"Lỗi: Phiên bản Python không tương thích.")
        print(f"Yêu cầu Python >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} và < {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print(f"Bạn đang dùng {current_version.major}.{current_version.minor}.")
        sys.exit(1)
    print(f"Phiên bản Python {current_version.major}.{current_version.minor} hợp lệ.")


def check_and_install_dependencies():
    """
    **Cải tiến:** Chạy `pip install -r requirements.txt` trực tiếp.
    Đây là cách làm "mượt" và đáng tin cậy nhất. `pip` đã được tối ưu
    để bỏ qua các gói đã cài đặt. Việc tự kiểm tra thủ công
    bằng `importlib.metadata` thường chậm và dễ lỗi hơn.
    """
    if not REQUIREMENTS_FILE.exists():
        print(f"Lỗi: Không tìm thấy file {REQUIREMENTS_FILE}.")
        print("Vui lòng đảm bảo file tồn tại trong thư mục gốc của dự án.")
        sys.exit(1)

    print("Đang đồng bộ hóa các thư viện từ requirements.txt...")
    try:
        # Sử dụng sys.executable để đảm bảo dùng đúng pip của môi trường python hiện tại
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE],
            stdout=subprocess.DEVNULL, # Ẩn output nếu thành công
            stderr=subprocess.PIPE
        )
        print("Tất cả thư viện cần thiết đã được cài đặt/cập nhật.")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi cài đặt thư viện:")
        print(e.stderr.decode('utf-8', errors='replace'))
        sys.exit(1)


def open_browser(delay_seconds: float = 5.0):
    """Mở trình duyệt web đến URL của web server sau một khoảng trễ."""
    try:
        with open(PORTS_FILE, "r", encoding="utf-8") as f:
            config = toml.load(f)
        
        port = config.get("webserver", {}).get("port", 8000)
        host = config.get("webserver", {}).get("host", "127.0.0.1")
        
        # Chuyển 0.0.0.0 thành 127.0.0.1 để mở trình duyệt
        if host == "0.0.0.0":
            host = "127.0.0.1"
            
        url = f"http://{host}:{port}/chat"
        print(f"Sẽ mở trình duyệt tại {url} sau {delay_seconds} giây...")
        
        # Sử dụng Timer để không block luồng chính
        threading.Timer(delay_seconds, lambda: webbrowser.open_new_tab(url)).start()
        
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy file {PORTS_FILE}, không thể tự động mở trình duyệt.")
    except Exception as e:
        print(f"Lỗi khi mở trình duyệt: {e}")


class ProcessManager:
    """
    Quản lý (khởi chạy, giám sát, khởi động lại, và tắt) các tiến trình con.
    """
    def __init__(self, config_file: Path):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Lỗi: File config '{config_file}' không tồn tại.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Lỗi: File config '{config_file}' không hợp lệ: {e}")
            sys.exit(1)
        
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.log_lock = threading.Lock()
        self.log_file_handle: Optional[Any] = None # Sẽ được mở trong hàm run()

    def start_process(self, name: str, cmd: List[str]):
        """
        Khởi động một tiến trình con, bắt output và tạo luồng log.
        **Cải tiến:** Thêm `creationflags` cho Windows để việc tắt
        tiến trình (shutdown) đáng tin cậy hơn.
        """
        print(f"Khởi động tiến trình [{name}]: {' '.join(cmd)}")
        
        # Cờ cho Windows để tạo process group mới, giúp proc.terminate()
        # gửi tín hiệu đến toàn bộ group, tương tự os.killpg trên Unix.
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                # Tạo process group mới trên Unix
                preexec_fn=os.setsid if sys.platform != "win32" else None,
                creationflags=creationflags # Cờ cho Windows
            )
            self.processes[name] = {'proc': proc, 'cmd': cmd}
            self._start_log_thread(name, proc)
        except Exception as e:
            print(f"Lỗi khi khởi động [{name}]: {e}")

    def _start_log_thread(self, name: str, proc: subprocess.Popen):
        """
        Tạo một luồng riêng để đọc output từ tiến trình con.
        **Cải tiến:** Sử dụng một file handle duy nhất (self.log_file_handle)
        và một Lock (self.log_lock) để đảm bảo thread-safe khi ghi log,
        thay vì mỗi luồng tự mở file.
        """
        def log_reader():
            if not self.log_file_handle:
                return
                
            # Đọc từng dòng output của tiến trình
            for line in iter(proc.stdout.readline, ''):
                if line.strip():
                    prefixed_line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{name}] {line.strip()}\n"
                    
                    # In ra console (print là thread-safe)
                    print(prefixed_line.strip())
                    
                    # Ghi vào file log (cần lock)
                    try:
                        with self.log_lock:
                            self.log_file_handle.write(prefixed_line)
                            self.log_file_handle.flush()
                    except Exception as e:
                        # Xử lý trường hợp file đã bị đóng
                        if "closed file" in str(e):
                            break
                        print(f"Lỗi ghi log từ [{name}]: {e}")
            
            # Đóng stdout của proc để giải phóng tài nguyên
            proc.stdout.close()

        t = threading.Thread(target=log_reader, daemon=True)
        t.start()
        self.processes[name]['log_thread'] = t

    def monitor_and_restart(self):
        """Vòng lặp vô hạn giám sát và khởi động lại tiến trình nếu bị lỗi."""
        while True:
            try:
                # Tạo bản copy của keys để tránh lỗi "dictionary changed size during iteration"
                for name in list(self.processes.keys()):
                    info = self.processes[name]
                    if info['proc'].poll() is not None:
                        print(f"Tiến trình [{name}] đã dừng (mã lỗi: {info['proc'].returncode}). Đang khởi động lại...")
                        self.start_process(name, info['cmd'])
                time.sleep(5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Lỗi trong vòng lặp giám sát: {e}")
                time.sleep(10) # Chờ lâu hơn nếu có lỗi

    def shutdown(self, signum=None, frame=None):
        """
        Tắt tất cả tiến trình con một cách an toàn.
        **Cải tiến:** Tắt theo thứ tự ngược lại, sử dụng terminate()
        (an toàn) trước, chờ (wait), rồi mới kill() (bắt buộc).
        """
        print("\nĐang tắt ứng dụng... Gửi tín hiệu dừng đến các tiến trình con.")
        
        # Tắt theo thứ tự ngược với lúc khởi động
        for name, info in reversed(self.processes.items()):
            proc = info['proc']
            if proc.poll() is None:  # Nếu tiến trình đang chạy
                print(f" - Đang dừng [{name}] (PID: {proc.pid})...", end='', flush=True)
                try:
                    # Gửi tín hiệu dừng
                    if sys.platform != "win32":
                        # Gửi SIGTERM đến toàn bộ process group
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        # Gửi CTRL_BREAK_EVENT (tương tự SIGTERM) đến process group
                        proc.terminate()
                    
                    # Chờ tiến trình tắt trong 5 giây
                    proc.wait(timeout=5)
                    print(" [OK]")

                except subprocess.TimeoutExpired:
                    # Nếu không tắt kịp, buộc dừng
                    print(" [Timeout! Buộc dừng]... ", end='', flush=True)
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill() # Tương tự taskkill /F
                    print("[KILLED]")
                except Exception as e:
                    print(f" [Lỗi: {e}]")
                    try:
                        proc.kill() # Cố gắng kill lần cuối
                    except:
                        pass # Bỏ qua nếu tiến trình đã chết
            else:
                 print(f" - Tiến trình [{name}] đã dừng từ trước.")

        # Đóng file log
        if self.log_file_handle:
            print("Đóng file log...")
            self.log_file_handle.close()
            
        print("Đã tắt xong. Tạm biệt!")
        sys.exit(0)

    def run(self):
        """Khởi chạy toàn bộ quy trình."""
        print_banner()
        check_python_version()
        check_and_install_dependencies()
        
        # Đăng ký tín hiệu shutdown (Ctrl+C và lệnh kill)
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        try:
            # **Cải tiến:** Mở file log một lần duy nhất
            self.log_file_handle = LOG_FILE.open('a', encoding='utf-8')
        except IOError as e:
            print(f"Lỗi: Không thể mở file log {LOG_FILE}: {e}")
            sys.exit(1)
            
        print("--- Khởi động các tiến trình con ---")
        for name, cmd in self.config.items():
            self.start_process(name, cmd)
        
        print("\n" + "="*40)
        print("Tất cả tiến trình đã được khởi động.")
        print(f"Log đang được ghi tại: {LOG_FILE}")
        print("Nhấn Ctrl+C để tắt ứng dụng một cách an toàn.")
        print("="*40 + "\n")
        
        open_browser()
        
        # Bắt đầu vòng lặp giám sát
        self.monitor_and_restart()


def main():
    """Hàm chính của script."""
    # **Cải tiến:** Đảm bảo script luôn chạy từ thư mục chứa nó
    os.chdir(BASE_DIR)
    
    # Tạo thư mục logs nếu chưa có
    LOG_DIR.mkdir(exist_ok=True)
    
    # **Cải tiến:** Xóa file log cũ mỗi khi khởi động
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except OSError as e:
        print(f"Cảnh báo: Không thể xóa file log cũ {LOG_FILE}: {e}")

    try:
        pm = ProcessManager(config_file=CONFIG_FILE)
        pm.run()
    except Exception as e:
        print(f"Lỗi nghiêm trọng không xác định: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()