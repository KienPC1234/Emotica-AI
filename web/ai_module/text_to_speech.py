import os
import logging
import json
from piper.voice import PiperVoice
from io import BytesIO
import wave
import time
from typing import Optional, Dict


try:
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False
    logging.warning("Thư viện langdetect chưa được cài đặt. Vui lòng chạy: pip install langdetect để hỗ trợ auto-detect ngôn ngữ.")


logging.getLogger("piper").setLevel(logging.WARNING)
log = logging.getLogger(__name__)



MODEL_DIR = os.path.join(os.path.dirname(__file__), "tts_models")


SUPPORTED_MODELS = {
    "vi": {
        "model": os.path.join(MODEL_DIR, "vi_VN-vais1000-medium.onnx"),
        "config": os.path.join(MODEL_DIR, "vi_VN-vais1000-medium.onnx.json"),
        "model_url": "https://huggingface.co/csukuangfj/vits-piper-vi_VN-vais1000-medium/resolve/main/vi_VN-vais1000-medium.onnx",
        "config_url": "https://huggingface.co/csukuangfj/vits-piper-vi_VN-vais1000-medium/resolve/main/vi_VN-vais1000-medium.onnx.json"
    },
    "en": {
        "model": os.path.join(MODEL_DIR, "en_US-joe-medium.onnx"),
        "config": os.path.join(MODEL_DIR, "en_US-joe-medium.onnx.json"),
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json"
    }
}


class TextToSpeech:
    """
    Lớp quản lý Text-to-Speech dùng piper-tts trên CPU.
    Tải các model Tiếng Việt (vi) và Tiếng Anh (en) khi khởi động.
    """
    def __init__(self):
        self.voices: Dict[str, PiperVoice] = {}
        self._load_models()

    def _load_models(self):
        """Tải tất cả các model được định nghĩa trong SUPPORTED_MODELS."""
        log.info("Đang tải các model piper-tts (trên CPU)...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        for lang_code, paths in SUPPORTED_MODELS.items():
            try:
                model_path = paths["model"]
                config_path = paths["config"]
                
                if not os.path.exists(model_path) or not os.path.exists(config_path):
                    log.warning(f"Không tìm thấy file model cho '{lang_code}' tại '{model_path}'.")
                    log.warning("Vui lòng tải model và đặt vào thư mục 'tts_models'.")
                    log.warning(f"Ví dụ lệnh tải (cho {lang_code}):")
                    log.warning(f"wget -O {model_path} {paths['model_url']}")
                    log.warning(f"wget -O {config_path} {paths['config_url']}")
                    continue
                
                start_time = time.time()
                
                
                voice = PiperVoice.load(model_path)
                
                
                self.voices[lang_code] = voice
                log.info(f"Đã tải xong model TTS '{lang_code}'. Thời gian: {time.time() - start_time:.2f}s")
                
            except Exception as e:
                log.error(f"Lỗi khi tải model TTS '{lang_code}': {e}", exc_info=True)

    def _detect_language(self, text: str) -> str:
        """Phát hiện ngôn ngữ tự động từ văn bản. Trả về 'vi' hoặc 'en', fallback về 'vi' nếu lỗi."""
        if not LANG_DETECT_AVAILABLE:
            log.warning("Không thể detect ngôn ngữ (langdetect chưa cài đặt). Fallback về 'vi'.")
            return "vi"
        
        try:
            detected_lang = detect(text)
            if detected_lang.startswith('vi'):
                return "vi"
            elif detected_lang.startswith('en'):
                return "en"
            else:
                log.warning(f"Ngôn ngữ phát hiện '{detected_lang}' không hỗ trợ. Fallback về 'vi'.")
                return "vi"
        except LangDetectException as e:
            log.warning(f"Lỗi detect ngôn ngữ: {e}. Fallback về 'vi'.")
            return "vi"

    def synthesize_speech(self, text: str, lang_code: str = "vi") -> Optional[bytes]:
        """
        Tổng hợp giọng nói từ văn bản.
        text: Văn bản cần chuyển thành giọng nói.
        lang_code: 'vi', 'en', hoặc 'auto' để tự động phát hiện.
        Trả về: Dữ liệu bytes thô của file WAV.
        """
        
        if lang_code == "auto":
            lang_code = self._detect_language(text)
            log.info(f"Tự động detect ngôn ngữ: '{lang_code}' cho văn bản: {text[:50]}...")
        
        if lang_code not in self.voices:
            log.error(f"Ngôn ngữ '{lang_code}' không được hỗ trợ hoặc model chưa được tải.")
            return None
        
        voice = self.voices[lang_code]
        
        try:
            log.debug(f"Bắt đầu tổng hợp TTS ({lang_code}): {text[:50]}...")
            
            
            audio_stream = BytesIO()
            with wave.open(audio_stream, 'wb') as wav_writer:
                voice.synthesize_wav(text, wav_writer)
            
            audio_bytes = audio_stream.getvalue()
            log.debug(f"Tổng hợp TTS thành công ({len(audio_bytes)} bytes).")
            return audio_bytes

        except Exception as e:
            log.error(f"Lỗi trong quá trình tổng hợp TTS: {e}", exc_info=True)
            return None


try:
    tts_analyzer = TextToSpeech()
except Exception as e:
    log.error(f"Không thể khởi tạo TextToSpeech: {e}")
    tts_analyzer = None


if __name__ == "__main__":
    import sys
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if tts_analyzer and (tts_analyzer.voices.get("vi") or tts_analyzer.voices.get("en")):
        log.info("--- Bắt đầu Test Module TextToSpeech ---")

        
        if "vi" in tts_analyzer.voices:
            log.info("--- Test: synthesize_speech (Tiếng Việt) ---")
            text_vi = "Xin chào, đây là thử nghiệm chuyển văn bản thành giọng nói."
            start_tts_vi = time.time()
            audio_vi = tts_analyzer.synthesize_speech(text_vi, lang_code="vi")
            
            if audio_vi:
                output_path_vi = "test_audio_vi.wav"
                with open(output_path_vi, "wb") as f:
                    f.write(audio_vi)
                log.info(f"===> Đã lưu file Tiếng Việt vào: {output_path_vi}")
                log.info(f"Thời gian tổng hợp (vi): {time.time() - start_tts_vi:.2f}s")
            else:
                log.error("Tổng hợp Tiếng Việt thất bại.")
        else:
            log.warning("Bỏ qua test Tiếng Việt (model chưa được tải).")

        
        if "en" in tts_analyzer.voices:
            log.info("\n--- Test: synthesize_speech (English) ---")
            text_en = "Hello, this is a test of the text to speech synthesis."
            start_tts_en = time.time()
            audio_en = tts_analyzer.synthesize_speech(text_en, lang_code="en")
            
            if audio_en:
                output_path_en = "test_audio_en.wav"
                with open(output_path_en, "wb") as f:
                    f.write(audio_en)
                log.info(f"===> Đã lưu file Tiếng Anh vào: {output_path_en}")
                log.info(f"Thời gian tổng hợp (en): {time.time() - start_tts_en:.2f}s")
            else:
                log.error("Tổng hợp Tiếng Anh thất bại.")
        else:
            log.warning("Bỏ qua test Tiếng Anh (model chưa được tải).")

        
        if LANG_DETECT_AVAILABLE:
            log.info("\n--- Test: synthesize_speech (Auto-Detect) ---")
            
            text_auto_en = "This is an English sentence for auto detection."
            start_tts_auto = time.time()
            audio_auto = tts_analyzer.synthesize_speech(text_auto_en, lang_code="auto")
            
            if audio_auto:
                output_path_auto = "test_audio_auto_en.wav"
                with open(output_path_auto, "wb") as f:
                    f.write(audio_auto)
                log.info(f"===> Đã lưu file Auto-Detect (English) vào: {output_path_auto}")
                log.info(f"Thời gian tổng hợp (auto): {time.time() - start_tts_auto:.2f}s")
            else:
                log.error("Tổng hợp Auto-Detect thất bại.")

            
            text_auto_vi = "Đây là câu tiếng Việt để kiểm tra tự động."
            start_tts_auto_vi = time.time()
            audio_auto_vi = tts_analyzer.synthesize_speech(text_auto_vi, lang_code="auto")
            
            if audio_auto_vi:
                output_path_auto_vi = "test_audio_auto_vi.wav"
                with open(output_path_auto_vi, "wb") as f:
                    f.write(audio_auto_vi)
                log.info(f"===> Đã lưu file Auto-Detect (Việt) vào: {output_path_auto_vi}")
                log.info(f"Thời gian tổng hợp (auto vi): {time.time() - start_tts_auto_vi:.2f}s")
            else:
                log.error("Tổng hợp Auto-Detect (Việt) thất bại.")
        else:
            log.warning("Bỏ qua test Auto-Detect (cần cài langdetect).")
            
        log.info("\n--- Test Module Hoàn Tất ---")
    else:
        log.error("Khởi tạo TTS Analyzer thất bại hoặc không tìm thấy model nào. Kết thúc test.")
        log.warning("Hãy chắc chắn bạn đã tải model vào thư mục 'tts_models' bằng lệnh wget ở trên.")