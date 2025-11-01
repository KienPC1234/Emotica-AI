import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM,
    BitsAndBytesConfig 
)
from io import BytesIO
import time
import logging
import base64
from typing import Union, Optional, Type

log = logging.getLogger(__name__)

logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)












class BaseImageCaptioner:
    """Lớp cơ sở trừu tượng cho các mô hình captioning."""
    def __init__(self, model_id: str, model_dir: str, device: str):
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.processor = None
        
        os.makedirs(self.model_dir, exist_ok=True)
        log.info(f"[Captioner] Chuẩn bị tải model: {self.model_id} vào {self.model_dir}")

    def load(self):
        """Tải model và processor."""
        raise NotImplementedError

    def generate_caption(self, image: Image.Image) -> str:
        """Tạo caption từ đối tượng PIL Image."""
        raise NotImplementedError

    @staticmethod
    def _preprocess_image(image_bytes: bytes, target_size: int = 2048) -> Image.Image:
        """
        Xử lý ảnh (bytes): resize nếu quá lớn, sau đó pad thành hình vuông.
        """
        try:
            
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = img.size
            log.debug(f"Ảnh gốc: {width}x{height}")

            
            if max(width, height) > target_size:
                scale = target_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                width, height = img.size 
                log.debug(f"Ảnh đã resize (do > {target_size}px): {width}x{height}")
            
            
            new_square_size = max(width, height)
            
            padded_img = Image.new('RGB', (new_square_size, new_square_size), (0, 0, 0))
            
            
            left = (new_square_size - width) // 2
            top = (new_square_size - height) // 2
            padded_img.paste(img, (left, top))
            
            log.debug(f"Ảnh đã được pad thành vuông: {new_square_size}x{new_square_size}")
            return padded_img
            
        except Exception as e:
            log.error(f"Lỗi khi tiền xử lý ảnh: {e}")
            raise

    @staticmethod
    def _postprocess_caption(caption: str) -> str:
        """Dọn dẹp câu chữ sau khi generate."""
        if not caption:
            return "Không thể tạo mô tả."
        caption = caption.strip()
        if caption:
            caption = caption[0].upper() + caption[1:]
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
        return caption



class BlipCaptioner(BaseImageCaptioner):
    def load(self):
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_id, cache_dir=self.model_dir, use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir,
                
                
                
                trust_remote_code=True
            ).to(self.device) 
            
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            log.info(f"[BlipCaptioner] Tải thành công {self.model_id} lên {self.device}")
        except Exception as e:
            log.error(f"[BlipCaptioner] Lỗi khi tải model: {e}")
            raise

    def generate_caption(self, image: Image.Image) -> str:
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return self._postprocess_caption(caption)



class GitCaptioner(BaseImageCaptioner):
    def load(self):
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                cache_dir=self.model_dir, 
                use_fast=True 
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir,
                
                
                
                trust_remote_code=True
            ).to(self.device) 
            
            log.info(f"[GitCaptioner] Tải thành công {self.model_id} lên {self.device}")
        except Exception as e:
            log.error(f"[GitCaptioner] Lỗi khi tải model: {e}")
            raise
    
    def generate_caption(self, image: Image.Image) -> str:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                max_length=128,
                num_beams=5
            )
        caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return self._postprocess_caption(caption)



class ImageCaptioningService:
    """
    Quản lý việc tải và chuyển đổi giữa các mô hình captioning.
    """
    
    MODEL_MAP = {
        "Salesforce/blip-image-captioning-large": BlipCaptioner,
        "microsoft/git-base-coco": GitCaptioner
    }
    
    def __init__(self, cache_root_dir="image_captioners"):
        self.device = "cpu"
        self.cache_root_dir = cache_root_dir
        self.current_model_id: Optional[str] = None
        self.captioner: Optional[BaseImageCaptioner] = None
        log.info(f"[ImageCaptioningService] Khởi tạo. Buộc chạy trên {self.device}.")

    def get_supported_models(self) -> list[str]:
        return list(self.MODEL_MAP.keys())

    def load_model(self, model_id: str):
        """Tải hoặc chuyển đổi sang một model captioning mới."""
        if model_id not in self.MODEL_MAP:
            raise ValueError(f"Model ID không được hỗ trợ: {model_id}")
        
        if self.current_model_id == model_id and self.captioner:
            log.info(f"Model {model_id} đã được tải.")
            return

        
        del self.captioner
        gc.collect() 
        
        model_class: Type[BaseImageCaptioner] = self.MODEL_MAP[model_id]
        model_dir = os.path.join(self.cache_root_dir, model_id.replace('/', '_'))
        
        log.info(f"Đang tải model mới: {model_id}")
        self.captioner = model_class(model_id, model_dir, self.device)
        self.captioner.load() 
        self.current_model_id = model_id
        log.info(f"Hoàn tất tải model: {model_id}")

    def caption(self, bytes_or_base64: Union[bytes, str]) -> str:
        """
        Tạo caption từ bytes hoặc chuỗi base64.
        """
        if not self.captioner:
            log.error("Lỗi: Không có model captioning nào được tải.")
            return "Lỗi: Dịch vụ captioning chưa sẵn sàng."

        try:
            image_bytes: bytes
            if isinstance(bytes_or_base64, str):
                
                if ',' in bytes_or_base64:
                    bytes_or_base64 = bytes_or_base64.split(',', 1)[-1]
                image_bytes = base64.b64decode(bytes_or_base64)
            else:
                image_bytes = bytes_or_base64
            
            
            processed_image = BaseImageCaptioner._preprocess_image(image_bytes)
            
            
            start_time = time.time()
            caption = self.captioner.generate_caption(processed_image)
            end_time = time.time()
            
            log.info(f"Caption (trong {end_time - start_time:.2f}s): {caption}")
            return caption
            
        except Exception as e:
            log.error(f"Lỗi nghiêm trọng khi tạo caption: {e}", exc_info=True)
            return f"Lỗi xử lý ảnh: {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    
    test_file = "test.jpg"
    img_bytes = None
    if not os.path.exists(test_file):
        log.warning(f"Không tìm thấy {test_file}, tạo ảnh dummy (màu đỏ).")
        dummy_img = Image.new('RGB', (800, 600), color='red') 
        img_byte_arr = BytesIO()
        dummy_img.save(img_byte_arr, format='JPEG', quality=95)
        img_bytes = img_byte_arr.getvalue()
    else:
        with open(test_file, 'rb') as f:
            img_bytes = f.read()

    
    service = ImageCaptioningService()
    
    
    print("\n=== Test với 'Salesforce/blip-image-captioning-large' ===")
    service.load_model("Salesforce/blip-image-captioning-large")
    caption1 = service.caption(img_bytes)
    print(f"Caption (BLIP): {caption1}")
    
    
    print("\n=== Test với 'microsoft/git-base-coco' ==T===")
    service.load_model("microsoft/git-base-coco")
    caption2 = service.caption(img_bytes)
    print(f"Caption (GIT): {caption2}")
    
    
    print("\n=== Test lại 'Salesforce/blip-image-captioning-large' ===")
    service.load_model("Salesforce/blip-image-captioning-large")
    caption3 = service.caption(img_bytes)
    print(f"Caption (BLIP 2): {caption3}")

    print("\n[INFO] Test ImageCaptioningService hoàn tất!")


