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


def rgb_gradient_text(text: str, start_rgb=(255, 0, 0), end_rgb=(0, 0, 255), steps=80):
    """Tạo gradient RGB thật mượt cho chuỗi text."""
    result = ""
    r1, g1, b1 = start_rgb
    r2, g2, b2 = end_rgb
    for i, ch in enumerate(text):
        ratio = i / max(len(text) - 1, 1)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        result += f"\033[38;2;{r};{g};{b}m{ch}\033[0m"
    return result


def print_banner():
    banner = r"""
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐  ███╗   ███╗ █████╗ ██╗  ██╗███████╗    ██████╗ ██╗   ██╗                   ▌
▐  ████╗ ████║██╔══██╗██║ ██╔╝██╔════╝    ██╔══██╗╚██╗ ██╔╝                   ▌
▐  ██╔████╔██║███████║█████╔╝ █████╗      ██████╔╝ ╚████╔╝                    ▌
▐  ██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝      ██╔══██╗  ╚██╔╝                     ▌
▐  ██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗    ██████╔╝   ██║                      ▌
▐  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝    ╚═╝                      ▌
▐                                                                             ▌
▐  ██╗  ██╗██╗███████╗███╗   ██╗██████╗  ██████╗ ██╗██████╗ ██████╗ ██╗  ██╗  ▌
▐  ██║ ██╔╝██║██╔════╝████╗  ██║██╔══██╗██╔════╝███║╚════██╗╚════██╗██║  ██║  ▌
▐  █████╔╝ ██║█████╗  ██╔██╗ ██║██████╔╝██║     ╚██║ █████╔╝ █████╔╝███████║  ▌
▐  ██╔═██╗ ██║██╔══╝  ██║╚██╗██║██╔═══╝ ██║      ██║██╔═══╝  ╚═══██╗╚════██║  ▌
▐  ██║  ██╗██║███████╗██║ ╚████║██║     ╚██████╗ ██║███████╗██████╔╝     ██║  ▌
▐  ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚═╝╚══════╝╚═════╝      ╚═╝  ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
https://github.com/KienPC1234
"""
    lines = banner.splitlines()
    colors = [
        (255, 0, 0),
        (255, 127, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 127, 255),
        (127, 0, 255),
        (255, 0, 255),
    ]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        print(rgb_gradient_text(line, start_rgb=color, end_rgb=colors[(i + 1) % len(colors)]))
        time.sleep(0.01)
    print("\n")


def check_python_version():
    """Kiểm tra phiên bản Python có nằm trong khoảng 3.10 đến 3.12 không."""
    min_version = (3, 10)
    max_version = (3, 13)
    current_version = sys.version_info
    if not (min_version <= current_version < max_version):
        print(f"Lỗi: Phiên bản Python không tương thích.")
        print(f"Yêu cầu Python 3.10, 3.11, hoặc 3.12. Bạn đang dùng {current_version.major}.{current_version.minor}.")
        sys.exit(1)
    print(f"Phiên bản Python {current_version.major}.{current_version.minor} hợp lệ.")


def check_and_install_dependencies():
    """Kiểm tra và cài đặt các thư viện bị thiếu từ requirements.txt."""
    requirements_file = 'requirements.txt'
    if not Path(requirements_file).exists():
        print(f"Lỗi: Không tìm thấy file {requirements_file}.")
        print("Vui lòng đảm bảo file tồn tại trong thư mục gốc của dự án.")
        sys.exit(1)

    print("Đang kiểm tra các thư viện cần thiết...")
    try:
        from importlib.metadata import distributions
    except ImportError:
        print("Lỗi: Không thể import `importlib.metadata`. Môi trường Python của bạn có thể bị lỗi.")
        sys.exit(1)

    installed_packages = {dist.metadata['name'].lower() for dist in distributions()}

    missing = []
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            req_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('[')[0].strip()
            if req_name and req_name.lower() not in installed_packages:
                missing.append(line)

    if missing:
        print(f"Phát hiện thiếu {len(missing)} thư viện. Đang tiến hành cài đặt...")
        for item in missing:
            print(f" - {item}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Cài đặt thành công.")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi cài đặt thư viện: {e}")
            sys.exit(1)
    else:
        print("Tất cả thư viện cần thiết đã được cài đặt.")


def open_browser():
    """Mở trình duyệt web đến URL của web server."""
    try:
        with open("ports.toml", "r") as f:
            config = toml.load(f)
        port = config.get("webserver", {}).get("port", 8000)
        host = config.get("webserver", {}).get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        url = f"http://{host}:{port}/chat"
        print(f"Sẽ mở trình duyệt tại {url} sau 5 giây...")
        threading.Timer(5.0, lambda: webbrowser.open_new_tab(url)).start()
    except FileNotFoundError:
        print("Cảnh báo: Không tìm thấy file ports.toml, không thể tự động mở trình duyệt.")
    except Exception as e:
        print(f"Lỗi khi mở trình duyệt: {e}")


class ProcessManager:
    def __init__(self, config_file='multiprocess.json'):
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Lỗi: File config '{config_file}' không tồn tại.")
            sys.exit(1)

        self.processes = {}
        self.log_file = Path('logs/latest.log')
        self.log_file.parent.mkdir(exist_ok=True)

    def start_process(self, name, cmd):
        print(f"Khởi động tiến trình [{name}]: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent.resolve()),  
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                preexec_fn=os.setsid if sys.platform != "win32" else None
            )
            self.processes[name] = {'proc': proc, 'cmd': cmd, 'log_thread': None}
            self._start_log_thread(name, proc)
        except Exception as e:
            print(f"Lỗi khi khởi động [{name}]: {e}")

    def _start_log_thread(self, name, proc):
        def log_reader():
            with self.log_file.open('a', encoding='utf-8') as log_f:
                for line in iter(proc.stdout.readline, ''):
                    if line.strip():
                        prefixed_line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{name}] {line.strip()}\n"
                        print(prefixed_line.strip())
                        log_f.write(prefixed_line)
                        log_f.flush()
        t = threading.Thread(target=log_reader, daemon=True)
        t.start()
        self.processes[name]['log_thread'] = t

    def monitor_and_restart(self):
        while True:
            for name, info in list(self.processes.items()):
                if info['proc'].poll() is not None:
                    print(f"Tiến trình [{name}] đã dừng. Đang khởi động lại...")
                    self.start_process(name, info['cmd'])
            time.sleep(5)

    def shutdown(self, signum=None, frame=None):
        print("\nĐang tắt ứng dụng... Gửi tín hiệu dừng đến các tiến trình con.")
        for name, info in self.processes.items():
            try:
                proc = info['proc']
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], check=True)
            except Exception:
                pass
        print("Đã tắt xong.")
        sys.exit(0)

    def run(self):
        print_banner()
        check_python_version()
        check_and_install_dependencies()
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        for name, cmd in self.config.items():
            self.start_process(name, cmd)
        print("\nTất cả tiến trình đã được khởi động.")
        open_browser()
        self.monitor_and_restart()


if __name__ == '__main__':
    
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    Path('logs').mkdir(exist_ok=True)
    pm = ProcessManager()
    pm.run()
