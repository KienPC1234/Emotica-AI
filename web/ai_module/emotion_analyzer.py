import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np


from deepface import DeepFace 
import logging
from io import BytesIO
import time
from typing import Optional



logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('deepface').setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)



def _bytes_to_cv2_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Chuyển đổi bytes ảnh thô sang frame OpenCV (np.ndarray)."""
    try:
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("Không thể decode image bytes. Dữ liệu có thể bị hỏng hoặc không phải là ảnh.")
            return None
        return img
    except Exception as e:
        log.error(f"Lỗi decode image bytes: {e}")
        return None



class EmotionAnalyzer:
    """
    Một lớp đơn giản để tải mô hình DeepFace một lần
    và cung cấp các hàm phân tích ảnh (trên CPU).
    """
    def __init__(self):
        self._pre_warm_model()

    def _pre_warm_model(self):
        """Tải mô hình DeepFace (trên CPU) lần đầu để tránh độ trễ."""
        try:
            log.info("Đang tải mô hình nhận diện cảm xúc (trên CPU)...")
            
            dummy_frame = np.zeros((48, 48, 3), dtype=np.uint8)
            
            
            
            DeepFace.analyze(
                dummy_frame, 
                actions=['emotion'], 
                enforce_detection=False
            )
            log.info("Mô hình đã tải xong (CPU).")
        except Exception as e:
            
            if "Face could not be detected" in str(e):
                log.info("Mô hình đã tải xong (CPU - bỏ qua lỗi 'không tìm thấy khuôn mặt').")
            else:
                log.warning(f"Lỗi khi tải mô hình: {e}")

    def get_emotion(self, image_bytes: bytes) -> str:
        """
        Nhận image_bytes, trả về cảm xúc chính (dominant_emotion).
        Trả về 'no_face' nếu không tìm thấy, 'error' nếu có lỗi.
        """
        img = _bytes_to_cv2_image(image_bytes)
        if img is None:
            return "error_decoding_image"
        
        try:
            
            result = DeepFace.analyze(
                img, 
                actions=['emotion'], 
                enforce_detection=True, 
                silent=True 
            )
            
            
            if isinstance(result, list):
                return result[0]['dominant_emotion'] 
            elif isinstance(result, dict):
                return result['dominant_emotion']
            else:
                return "unknown_result"
                
        except Exception as e:
            
            if "Face could not be detected" in str(e):
                return "no_face"
            else:
                log.error(f"Lỗi khi phân tích cảm xúc: {e}")
                return "error"

    def test_draw_emotion(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Nhận image_bytes, vẽ bounding box + cảm xúc lên TẤT CẢ các khuôn mặt,
        trả về image_bytes mới. Trả về None nếu có lỗi nghiêm trọng.
        """
        img = _bytes_to_cv2_image(image_bytes)
        if img is None:
            return None
        
        try:
            
            
            results = DeepFace.analyze(
                img, 
                actions=['emotion'], 
                enforce_detection=False, 
                silent=True
            )
            
            
            if not isinstance(results, list):
                 results = [results] 
                 
            found_faces = 0
            for result in results:
                
                if 'region' not in result or 'dominant_emotion' not in result:
                    continue
                
                found_faces += 1
                r = result['region'] 
                emotion = result['dominant_emotion']
                
                
                
                box_color = (0, 255, 0)
                cv2.rectangle(
                    img, 
                    (r['x'], r['y']), 
                    (r['x'] + r['w'], r['y'] + r['h']), 
                    box_color, 
                    2 
                )
                
                
                text = f"{emotion}"
                text_color = (0, 0, 0) 
                
                
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                
                text_x = r['x']
                text_y = r['y'] - 10
                
                
                cv2.rectangle(
                    img, 
                    (text_x, text_y - text_h - 5), 
                    (text_x + text_w + 10, text_y + 5), 
                    box_color, 
                    -1 
                )
                
                
                cv2.putText(
                    img, 
                    text, 
                    (text_x + 5, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    text_color, 
                    2
                )
            
            if found_faces == 0:
                log.warning("Không tìm thấy khuôn mặt nào để vẽ.")
            
            
            is_success, buffer = cv2.imencode('.jpg', img)
            if not is_success:
                log.error("Không thể encode ảnh đầu ra.")
                return None
            
            return buffer.tobytes()

        except Exception as e:
            log.error(f"Lỗi khi vẽ cảm xúc: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    import sys
    
    
    log.info("--- Bắt đầu Test Module EmotionAnalyzer ---")
    start_load = time.time()
    analyzer = EmotionAnalyzer() 
    log.info(f"Thời gian tải model (CPU): {time.time() - start_load:.2f}s")
    
    
    test_image_path = "test.jpg" 
    if not os.path.exists(test_image_path):
        log.error(f"Không tìm thấy ảnh test '{test_image_path}'. Vui lòng đặt ảnh vào cùng thư mục.")
        sys.exit()
        
    log.info(f"Đang đọc ảnh test: {test_image_path}")
    with open(test_image_path, "rb") as f:
        image_bytes_data = f.read()
    
    
    log.info(f"--- Test: get_emotion ---")
    start_analyze = time.time()
    emotion = analyzer.get_emotion(image_bytes_data)
    log.info(f"===> Cảm xúc phát hiện: {emotion}")
    log.info(f"Thời gian phân tích: {time.time() - start_analyze:.2f}s")
    
    
    log.info(f"--- Test: test_draw_emotion ---")
    start_draw = time.time()
    drawn_image_bytes = analyzer.test_draw_emotion(image_bytes_data)
    
    if drawn_image_bytes:
        output_path = "test_image_output.jpg"
        with open(output_path, "wb") as f:
            f.write(drawn_image_bytes)
        log.info(f"===> Đã vẽ và lưu kết quả vào: {output_path}")
        log.info(f"Thời gian vẽ: {time.time() - start_draw:.2f}s")
    else:
        log.error("Test vẽ thất bại.")
        
    log.info("--- Test Module Hoàn Tất ---")
