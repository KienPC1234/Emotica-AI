import os
import logging
from faster_whisper import WhisperModel
from io import BytesIO
import time
from typing import Optional, BinaryIO


logging.getLogger("faster_whisper").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

class SpeechToText:
    """
    Lớp quản lý Speech-to-Text dùng faster-whisper trên CPU.
    Tải model 'base' một lần.
    """
    def __init__(self, model_size="base"):
        self.model = None
        try:
            log.info(f"Đang tải model faster-whisper '{model_size}' (trên CPU)...")
            start_time = time.time()
            
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            log.info(f"Đã tải xong model faster-whisper. Thời gian: {time.time() - start_time:.2f}s")
        except Exception as e:
            log.error(f"Lỗi nghiêm trọng khi tải model faster-whisper: {e}", exc_info=True)

    def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """
        Phiên âm audio từ bytes.
        audio_bytes: Dữ liệu bytes thô của một file audio (WAV, MP3, M4A, v.v.)
        Trả về: Văn bản đã phiên âm hoặc None nếu lỗi.
        """
        if not self.model:
            log.error("Model faster-whisper chưa được tải. Không thể phiên âm.")
            return None

        try:
            log.debug("Bắt đầu phiên âm audio bytes...")
            
            audio_input = BytesIO(audio_bytes)
            
            
            
            segments, info = self.model.transcribe(
                audio_input,
                beam_size=5,
                language=None 
            )

            log.info(f"Phát hiện ngôn ngữ: {info.language} (xác suất {info.language_probability:.2f})")

            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            
            log.debug(f"Văn bản phiên âm: {full_text.strip()}")
            return full_text.strip()

        except Exception as e:
            log.error(f"Lỗi trong quá trình phiên âm: {e}", exc_info=True)
            return None



try:
    stt_analyzer = SpeechToText()
except Exception as e:
    log.error(f"Không thể khởi tạo SpeechToText: {e}")
    stt_analyzer = None


if __name__ == "__main__":
    import sys
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if stt_analyzer:
        log.info("--- Bắt đầu Test Module SpeechToText ---")
        
        test_audio_path = "test_audio.wav" 
        if not os.path.exists(test_audio_path):
            log.error(f"Không tìm thấy file audio test '{test_audio_path}'.")
            sys.exit()
            
        log.info(f"Đang đọc file audio test: {test_audio_path}")
        with open(test_audio_path, "rb") as f:
            audio_bytes_data = f.read()
            
        log.info("--- Test: transcribe_audio ---")
        start_transcribe = time.time()
        transcription = stt_analyzer.transcribe_audio(audio_bytes_data)
        
        if transcription:
            log.info(f"===> Văn bản: {transcription}")
            log.info(f"Thời gian phiên âm: {time.time() - start_transcribe:.2f}s")
        else:
            log.error("Phiên âm thất bại.")
            
        log.info("--- Test Module Hoàn Tất ---")
    else:
        log.error("Khởi tạo STT Analyzer thất bại. Kết thúc test.")
