#!/bin/bash
# Kịch bản khởi động cho Emotica AI trên Linux và macOS

# Dừng tất cả các tiến trình con khi thoát
trap 'kill $(jobs -p) &>/dev/null' EXIT

echo "--- Bắt đầu khởi động Emotica AI ---"

# 1. Kiểm tra phiên bản Python
echo "[1/4] Kiểm tra phiên bản Python..."
python3 -c 'import sys; exit(0) if (3, 10) <= sys.version_info < (3, 13) else sys.exit(1)'
if [ $? -ne 0 ]; then
    echo "Lỗi: Yêu cầu Python 3.10, 3.11 hoặc 3.12."
    exit 1
fi
echo "Phiên bản Python hợp lệ."

# 2. Cài đặt thư viện
echo "[2/4] Kiểm tra và cài đặt các thư viện..."
pip3 install -r requirements.txt
echo "Các thư viện đã được cài đặt."

# 3. Khởi động các server
echo "[3/4] Khởi động Llama Server và Web Server..."
python3 llama_server.py &
python3 web/main.py &

# 4. Mở trình duyệt
echo "[4/4] Đợi server khởi động để mở trình duyệt..."
sleep 5

# Đọc cổng từ file ports.toml, nếu không có thì dùng 8000
PORT=$(grep -oP 'port\s*=\s*\K[0-9]+' ports.toml | head -n 1)
URL="http://127.0.0.1:${PORT:-8000}/chat"

echo "Đang mở ứng dụng tại: $URL"
# Thử các lệnh khác nhau để mở trình duyệt
xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || echo "Không thể tự mở trình duyệt. Vui lòng truy cập thủ công: $URL"

echo "--- Emotica AI đã sẵn sàng ---"
echo "Nhấn Ctrl+C trong terminal này để tắt ứng dụng."

# Đợi các tiến trình nền kết thúc
wait
