from image2text import ImageCaptioningService
from rag_manager import rag_manager

import os
import gc
import toml
import logging
import asyncio
import json
import base64
import psutil 
import platform 
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import uvicorn
import torch 
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse 
from pydantic import BaseModel, Field
from llama_cpp import Llama


log = logging.getLogger(__name__)

try:
    import pynvml 
    pynvml.nvmlInit()
    GPU_MONITORING_ENABLED = True
except ImportError:
    GPU_MONITORING_ENABLED = False
except Exception as e:
    GPU_MONITORING_ENABLED = False
    log.warning(f"Lỗi khi khởi tạo pynvml: {e}. Sẽ giám sát CPU/RAM.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODELS_DIR = Path("models")
CONFIG_FILE = MODELS_DIR / "cfg.json"


class LlamaModelConfig(BaseModel):
    name: str = Field(..., description="Tên hiển thị của model, ví dụ: 'Llama-3-8B-Instruct'")
    path: str = Field(..., description="Đường dẫn đầy đủ tới file .gguf, ví dụ: 'models/llama-3-8b.Q4_K_M.gguf'")
    format: str = Field(..., description="Định dạng chat handler, ví dụ: 'llama-3' hoặc 'mistral-instruct'")
    description: Optional[str] = Field(None, description="Mô tả ngắn gọn về model")
    max_context: int = Field(4096, description="Kích thước context (n_ctx) tối đa cho model")
    temperature: float = Field(0.7, description="Nhiệt độ mặc định cho generation")
    top_p: float = Field(0.9, description="Top-p mặc định cho generation")


class LlamaManager:
    def __init__(self, models_dir: Path, config_file: Path):
        self.models_dir = models_dir
        self.config_file = config_file
        self.llama_instance: Optional[Llama] = None
        self.current_model_config: Optional[LlamaModelConfig] = None
        
        self.model_configs: Dict[str, LlamaModelConfig] = {}
        self.lock = asyncio.Lock() 
        self.reload_config_sync()

    def reload_config_sync(self):
        
        if not self.config_file.exists():
            log.warning(f"Không tìm thấy {self.config_file}. Tạo file mẫu.")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            sample_config = {
                "models": [
                    {
                        "name": "Llama-3-8B-Instruct-Demo",
                        "path": "models/llama-3-8b-instruct.Q4_K_M.gguf",
                        "format": "llama-3",
                        "description": "Model LLaMA 3 8B (Demo - Vui lòng thay thế đường dẫn)",
                        "max_context": 4096,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                ],
                "valid_formats": ["llama-3", "mistral-instruct", "chatml"] 
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            self.model_configs = {}
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            
            configs_list = data.get("models", []) 
            
            temp_configs = {}
            for cfg_dict in configs_list:
                try:
                    
                    model_cfg = LlamaModelConfig(**cfg_dict)
                    
                    temp_configs[model_cfg.name] = model_cfg
                except Exception as e:
                    log.warning(f"Bỏ qua config model không hợp lệ: {cfg_dict}. Lỗi: {e}")

            self.model_configs = temp_configs
            log.info(f"Đã tải {len(self.model_configs)} cấu hình model LLaMA từ {self.config_file}")
            
        except Exception as e:
            log.error(f"Lỗi khi đọc {self.config_file}: {e}", exc_info=True)
            self.model_configs = {}

    def get_available_models(self) -> List[Dict[str, Any]]:
        
        return [
            {
                "name": cfg.name,
                "path": cfg.path,
                "format": cfg.format,
                "description": cfg.description,
                "max_context": cfg.max_context
            }
            for cfg in self.model_configs.values()
        ]

    def unload_model_sync(self):
        
        if self.llama_instance:
            log.info(f"Đang giải phóng model LLaMA: {self.current_model_config.name}")
            del self.llama_instance
            self.llama_instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
        self.current_model_config = None

    def load_model_sync(self, model_name: str):
        
        if model_name not in self.model_configs:
            raise FileNotFoundError(f"Không tìm thấy cấu hình cho model '{model_name}' trong {self.config_file}")
        
        config = self.model_configs[model_name]
        
        model_path = Path(config.path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"File model GGUF không tồn tại tại đường dẫn: {model_path}")

        
        if self.current_model_config and self.current_model_config.name == model_name and self.llama_instance:
            log.info(f"Model LLaMA '{config.name}' đã được tải.")
            return

        
        self.unload_model_sync()
        
        log.info(f"Đang tải model LLaMA: {config.name} (Handler: {config.format})...")
        try:
            self.llama_instance = Llama(
                model_path=str(model_path),
                n_gpu_layers=-1, 
                n_ctx=config.max_context, 
                chat_format=config.format, 
                verbose=False
            )
            self.current_model_config = config
            log.info(f"Tải model LLaMA thành công: {config.name}")
        except Exception as e:
            log.error(f"Lỗi khi tải model LLaMA {config.name}: {e}", exc_info=True)
            self.current_model_config = None
            self.llama_instance = None
            raise


app = FastAPI(
    title="LLaMA-CPP + Vision + RAG API Server",
    description="API server tích hợp LLaMA (Text), Vision (Captioning) và RAG (Vector DB)",
    version="2.4.0" 
)

llama_manager = LlamaManager(models_dir=MODELS_DIR, config_file=CONFIG_FILE)
caption_manager = ImageCaptioningService(cache_root_dir="image_captioners")
caption_lock = asyncio.Lock() 


def get_system_stats_sync():
    stats = {"gpu_monitoring_enabled": GPU_MONITORING_ENABLED}
    
    if GPU_MONITORING_ENABLED:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            stats.update({
                "system_type": "gpu",
                "gpu_model": pynvml.nvmlDeviceGetName(handle),
                "cuda_version": pynvml.nvmlSystemGetDriverVersion(),
                "vram_total_gb": round(mem_info.total / (1024**3), 2),
                "vram_free_gb": round(mem_info.free / (1024**3), 2),
                "vram_used_gb": round(mem_info.used / (1024**3), 2),
                "gpu_load_percent": util_rates.gpu,
                "vram_load_percent": round((mem_info.used / mem_info.total) * 100, 2)
            })
        except Exception as e:
            log.warning(f"Lỗi khi lấy thông tin pynvml: {e}")
            stats["system_type"] = "gpu_error"
            stats["error"] = str(e)
    else:
        cpu_load = psutil.cpu_percent()
        ram_info = psutil.virtual_memory()
        stats.update({
            "system_type": "cpu",
            "cpu_model": platform.processor(),
            "cpu_load_percent": cpu_load,
            "ram_total_gb": round(ram_info.total / (1024**3), 2),
            "ram_free_gb": round(ram_info.available / (1024**3), 2),
            "ram_used_gb": round(ram_info.used / (1024**3), 2),
            "ram_load_percent": ram_info.percent
        })
    return stats


@app.on_event("startup")
async def startup_event():
    log.info("Server đang khởi động...")
    
    if not rag_manager:
        log.error("Khởi tạo RAG Manager thất bại. Các tính năng RAG sẽ không hoạt động.")
    
    
    llama_manager.reload_config_sync()
    models = llama_manager.get_available_models()
    if models:
        
        default_llama_name = models[0]["name"]
        log.info(f"Đang tải LLaMA model mặc định: {default_llama_name}")
        try:
            
            await asyncio.to_thread(llama_manager.load_model_sync, default_llama_name)
        except Exception as e:
            log.error(f"Lỗi tải LLaMA model mặc định: {e}")
    else:
        log.warning("Không tìm thấy model LLaMA nào trong cfg.json.")
        
    
    caption_models = caption_manager.get_supported_models()
    if caption_models:
        default_captioner = "Salesforce/blip-image-captioning-large" 
        log.info(f"Đang tải Captioner model mặc định: {default_captioner}")
        async with caption_lock:
            try:
                await asyncio.to_thread(caption_manager.load_model, default_captioner)
            except Exception as e:
                log.error(f"Lỗi tải Captioner model mặc định: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    log.info("Server đang tắt...")
    if GPU_MONITORING_ENABLED:
        try:
            pynvml.nvmlShutdown()
            log.info("Đã tắt pynvml.")
        except Exception as e:
            log.warning(f"Lỗi khi tắt pynvml: {e}")


class SwitchLlamaRequest(BaseModel):
    model_name: str = Field(..., description="Tên hiển thị của model (từ /models) muốn chuyển, ví dụ: 'Llama-3-8B-Instruct'")

@app.get("/models", tags=["LLaMA (Text) Management"])
async def get_llama_models():
    
    return {"models": llama_manager.get_available_models()}

@app.get("/models/current", tags=["LLaMA (Text) Management"])
async def get_current_llama_model():
    
    async with llama_manager.lock:
        name = llama_manager.current_model_config.name if llama_manager.current_model_config else None
        path = llama_manager.current_model_config.path if llama_manager.current_model_config else None
    return {"current_model_name": name, "current_model_path": path}

@app.post("/models/switch", tags=["LLaMA (Text) Management"])
async def switch_llama_model(request: SwitchLlamaRequest):
    
    log.info(f"Yêu cầu chuyển LLaMA model sang: {request.model_name}")
    async with llama_manager.lock:
        try:
            
            await asyncio.to_thread(llama_manager.load_model_sync, request.model_name)
            return {"status": "success", "loaded_model_name": request.model_name}
        except Exception as e:
            log.error(f"Lỗi khi chuyển LLaMA model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/reload-config", tags=["LLaMA (Text) Management"])
async def reload_llama_config():
    
    log.info("Yêu cầu tải lại file cfg.json...")
    async with llama_manager.lock:
        try:
            await asyncio.to_thread(llama_manager.reload_config_sync)
            return {
                "status": "success", 
                "message": "Đã tải lại cfg.json. Gọi /models để xem danh sách mới.",
                "available_models": [cfg.name for cfg in llama_manager.model_configs.values()]
            }
        except Exception as e:
            log.error(f"Lỗi khi tải lại cfg.json: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


class SwitchCaptionerRequest(BaseModel):
    model_id: str

@app.get("/captioner/models", tags=["Captioner (Vision) Management"])
async def get_captioner_models():
    return {"supported_models": caption_manager.get_supported_models()}

@app.get("/captioner/current", tags=["Captioner (Vision) Management"])
async def get_current_captioner_model():
    async with caption_lock:
        model_id = caption_manager.current_model_id
    return {"current_model_id": model_id}

@app.post("/captioner/switch", tags=["Captioner (Vision) Management"])
async def switch_captioner_model(request: SwitchCaptionerRequest):
    async with caption_lock:
        try:
            await asyncio.to_thread(caption_manager.load_model, request.model_id)
            return {"status": "success", "loaded_model_id": request.model_id}
        except Exception as e:
            log.error(f"Lỗi khi chuyển Captioner model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/stats", tags=["System & Monitoring"])
async def get_system_stats():
    
    try:
        stats = await asyncio.to_thread(get_system_stats_sync)
        return JSONResponse(content=stats)
    except Exception as e:
        log.error(f"Lỗi khi lấy thông tin hệ thống: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/upload", tags=["RAG (Vector DB) Management"])
async def upload_rag_document(file: UploadFile = File(...)):
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG service is not available.")
        
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    allowed_exts = {'.pdf', '.txt', '.md'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {allowed_exts}")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        log.info(f"Đã nhận file RAG: {file.filename}. Đang xử lý...")
        
        success = await asyncio.to_thread(
            rag_manager.add_document, 
            temp_file_path, 
            file.filename
        )
        
        if success:
            return {"status": "success", "filename": file.filename, "message": "Đã thêm và lập chỉ mục (indexed)."}
        else:
            raise HTTPException(status_code=500, detail="Lỗi khi xử lý và lập chỉ mục file.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi upload: {e}")
    finally:
        if file:
            file.file.close()
        if temp_file_path.exists():
            temp_file_path.unlink() 

@app.get("/rag/list", tags=["RAG (Vector DB) Management"])
async def list_rag_documents():
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG service is not available.")
    
    try:
        docs = await asyncio.to_thread(rag_manager.list_documents)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rag/delete", tags=["RAG (Vector DB) Management"])
async def delete_rag_document(filename: str):
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG service is not available.")
        
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
        
    try:
        success = await asyncio.to_thread(rag_manager.delete_document, filename)
        if success:
            return {"status": "success", "filename": filename, "message": "Đã xóa khỏi RAG DB."}
        else:
            raise HTTPException(status_code=404, detail=f"File '{filename}' không tìm thấy trong RAG DB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




async def preprocess_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_messages = []
    
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            
            processed_messages.append(message)
            continue
        
        
        new_content = ""
        for part in content:
            if part.get("type") == "text":
                new_content += part.get("text", "") + "\n"
            
            elif part.get("type") == "image_url":
                image_data_url = part.get("image_url", {}).get("url", "")
                if not image_data_url.startswith("data:image"):
                    log.warning(f"Bỏ qua image_url không hợp lệ: {image_data_url[:50]}...")
                    new_content += "[Invalid image URL]\n"
                    continue
                
                log.info("Phát hiện hình ảnh trong prompt. Đang tạo caption...")
                try:
                    base64_data = image_data_url.split(',', 1)[-1]
                    async with caption_lock:
                        if not caption_manager.captioner:
                            log.error("Captioner chưa sẵn sàng!")
                            caption = "[Error: Image description service unavailable]"
                        else:
                            caption = await asyncio.to_thread(caption_manager.caption, base64_data)
                    new_content += f"[Image description: {caption}]\n"
                except Exception as e:
                    log.error(f"Lỗi khi caption ảnh: {e}")
                    new_content += "[Error while processing image]\n"
        
        
        message["content"] = new_content.strip()
        processed_messages.append(message)
    
    return processed_messages




async def preprocess_rag(messages: List[Dict[str, Any]], use_rag: bool) -> List[Dict[str, Any]]:
    if not use_rag or not rag_manager:
        return messages

    log.info("Đang thực hiện RAG...")

    
    last_user_query = ""
    last_user_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_query = messages[i].get("content", "").strip()
            last_user_index = i
            break

    if not last_user_query:
        log.warning("RAG: Không tìm thấy tin nhắn user để làm query.")
        return messages

    
    log.debug(f"RAG Query: {last_user_query}")
    context_chunks = await asyncio.to_thread(rag_manager.search, last_user_query)

    if not context_chunks:
        log.warning("RAG: Không tìm thấy context nào.")
        return messages  

    
    context_str = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
    sources = list(set([chunk['source'] for chunk in context_chunks]))

    log.info(f"RAG: Đã tìm thấy context từ: {sources}")

    
    rag_context_block = (
    "\n\n--- CONTEXT INFORMATION ---\n"
    "Here is some reference information (context) that may be useful for the above question:\n"
    f"{context_str}\n"
    "--- END CONTEXT ---\n\n"
    "→ Based on the above content, please answer accurately and naturally."
    )

    
    if last_user_index != -1:
        original_content = messages[last_user_index].get("content", "")
        if not isinstance(original_content, str):
            original_content = str(original_content)
        messages[last_user_index]["content"] = original_content + rag_context_block
        log.info("Đã nối context RAG vào tin nhắn user cuối cùng.")
    else:
        
        messages.append({"role": "user", "content": f"Question context:\n{rag_context_block}"})
        log.warning("RAG: Không tìm thấy user message, đã tạo mới với context RAG.")

    return messages






@app.post("/v1/chat/completions", tags=["Inference"])
async def create_chat_completion(request: Request):
    
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    messages = body.get("messages", [])
    use_rag = body.get("use_rag", False) 
    
    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' field.")

    async with llama_manager.lock:
        if llama_manager.llama_instance is None:
            raise HTTPException(status_code=503, detail="Không có model LLaMA nào đang được tải.")

        try:
            
            model_defaults = llama_manager.current_model_config
            
            
            stream = body.get("stream", False)
            max_tokens = body.get("max_tokens", 1024) 
            temperature = body.get("temperature", model_defaults.temperature)
            top_p = body.get("top_p", model_defaults.top_p)
            
            log.debug(f"Params (Final): temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")

            
            
            processed_messages_vision = await preprocess_messages(messages)
            
            
            
            final_messages = await preprocess_rag(processed_messages_vision, use_rag)
            
            
            def run_llama_inference():
                log.debug(f"Gửi tới LLaMA: {final_messages}")
                return llama_manager.llama_instance.create_chat_completion(
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=temperature, 
                    top_p=top_p, 
                    stream=stream
                )

            
            completion_or_streamer = await asyncio.to_thread(run_llama_inference)

            
            if stream:
                def stream_generator(streamer):
                    try:
                        for chunk in streamer:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception as e:
                        log.error(f"Lỗi trong quá trình stream LLaMA: {e}")
                    finally:
                        log.debug("Stream LLaMA hoàn tất.")
                
                return StreamingResponse(
                    stream_generator(completion_or_streamer),
                    media_type="text/event-stream"
                )
            else:
                return completion_or_streamer

        except Exception as e:
            log.error(f"Lỗi trong quá trình inference: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Lỗi inference: {e}")


def load_config():
    DEFAULT_PORT = 8002
    try:
        with open("ports.toml", "r") as f:
            config = toml.load(f)
        port = config.get("llama_server", {}).get("port", DEFAULT_PORT)
        return port
    except Exception:
        log.warning(f"Không tìm thấy ports.toml, dùng cổng mặc định {DEFAULT_PORT}")
        return DEFAULT_PORT

if __name__ == "__main__":
    port = load_config()
    log.info(f"--- Khởi chạy LLaMA-CPP + Vision + RAG Server trên http://0.0.0.0:{port} ---")
    log.info(f"--- Xem tài liệu API tại http://127.0.0.1:{port}/docs ---")
    uvicorn.run(app, host="0.0.0.0", port=port)