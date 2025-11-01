import os
import toml
import json
import httpx
import uvicorn
import logging
import base64
import shutil
import re
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, AsyncGenerator

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import sessionmaker, relationship, DeclarativeBase
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ai_module.emotion_analyzer import EmotionAnalyzer
from ai_module.speech_to_text import stt_analyzer
from ai_module.text_to_speech import tts_analyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

try:
    with open("ports.toml", "r") as f:
        config = toml.load(f)
    web_config = config.get("webserver", {})
    WEB_SERVER_HOST = web_config.get("host", "0.0.0.0")
    WEB_SERVER_PORT = web_config.get("port", 8000)
    
    LLAMA_SERVER_PORT = config.get("llama_server", {}).get("port", 8002)
except Exception:
    WEB_SERVER_HOST = "0.0.0.0"
    WEB_SERVER_PORT = 8000
    LLAMA_SERVER_PORT = 8002

LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"
log.info(f"Web server sẽ chạy tại http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
log.info(f"Kết nối đến Llama Server tại {LLAMA_SERVER_URL}")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

STATIC_DIR = BASE_DIR / "statics"
TEMPLATES_DIR = BASE_DIR / "templates"
TEMP_IMAGE_DIR = BASE_DIR / "temp_user_image"
LLAMA_CONFIG_FILE = ROOT_DIR / "models" / "cfg.json"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
TEMP_IMAGE_DIR.mkdir(exist_ok=True)
(ROOT_DIR / "models").mkdir(exist_ok=True) 

DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String, index=True) 
    content = Column(Text)
    image_url = Column(String, nullable=True) 
    timestamp = Column(DateTime, default=datetime.now)
    session = relationship("ChatSession", back_populates="messages")

class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class MemoryUpdate(BaseModel):
    content: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Emotica AI Web Server")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/temp_user_image", StaticFiles(directory=TEMP_IMAGE_DIR), name="temp_images")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    emotion_service = EmotionAnalyzer()
    log.info("Emotion Analyzer (deepface) đã tải xong trên CPU.")
except Exception as e:
    log.error(f"Lỗi khởi tạo EmotionAnalyzer: {e}", exc_info=True)
    emotion_service = None

if stt_analyzer: log.info("SpeechToText (faster-whisper) đã tải xong.")
if tts_analyzer: log.info("TextToSpeech (piper) đã tải xong.")

http_client = httpx.AsyncClient(timeout=None)





USER_MEMORY_INSTRUCTIONS = (
    "\n--- USER MEMORY INSTRUCTIONS ---\n"
    "You have the ability to remember important, long-term information about the user (e.g., name, hobbies, goals, major life events they share) to be more understanding and empathetic."
    "To save a piece of information, YOU MUST PLACE IT IN A SPECIAL TAG <saveUserInfo> AT THE VERY BEGINNING OF YOUR RESPONSE. This tag will be hidden and not shown to the user."
    "Example: <saveUserInfo>User's name is An, and they like Python programming.</saveUserInfo>Hello An, I'd be happy to talk about Python..."
    "Only save core facts, not fleeting emotions or casual small talk."
    "You can also access saved information (if any) in the [USER MEMORIES] section below."
    "Please use these memories to personalize the conversation and show empathy."
    "TAG <saveUserInfo> AT THE BEGINNING OF YOUR RESPONSE!"
)

EMOTICA_SYSTEM_PROMPT = (
    "You are Emotica AI, a compassionate and therapeutic virtual assistant. "
    "Your primary goal is to listen, understand, and provide empathetic, supportive, and non-judgmental responses. "
    "The user may be sharing their feelings, and your role is to be a safe space for them. "
    "If the user provides their current emotion (e.g., 'User is feeling: sad') or an emotional journey (e.g., 'User's emotional journey: [list, of, emotions]'), "
    "acknowledge this information with care and tailor your response to their emotional state. Always be kind, patient, and encouraging."
    f"{USER_MEMORY_INSTRUCTIONS}"
)

def get_user_memory_block(db: Session) -> str:
    try:
        memories = db.query(UserMemory).order_by(UserMemory.created_at.desc()).all()
        if not memories:
            return "\n[USER MEMORIES: (None yet)]\n"
        
        memory_list = [f"- {m.content} (Saved: {m.created_at.strftime('%Y-%m-%d')})" for m in memories]
        return "\n[USER MEMORIES:]\n" + "\n".join(memory_list) + "\n---"
    except Exception as e:
        log.error(f"Lỗi khi truy vấn UserMemory: {e}")
        return "\n[USER MEMORIES: (Error loading memories)]\n"





async def save_user_memory_task(memory_content: str, db_session: Session):
    try:
        current_count = db_session.query(func.count(UserMemory.id)).scalar()
        
        if current_count >= 100:
            log.warning(f"UserMemory đã đạt giới hạn (>= 100). Không thêm: {memory_content}")
        
        new_memory = UserMemory(content=memory_content.strip())
        db_session.add(new_memory)
        db_session.commit()
        log.info(f"Đã lưu vào UserMemory: {memory_content}")
        
    except Exception as e:
        log.error(f"Lỗi khi lưu UserMemory (task): {e}", exc_info=True)
        db_session.rollback()
    finally:
        db_session.close()

@app.on_event("startup")
async def on_startup():
    Base.metadata.create_all(bind=engine)
    
    global http_client
    http_client = httpx.AsyncClient(timeout=None)
    log.info("HTTP client (httpx) đã sẵn sàng.")

@app.on_event("shutdown")
async def on_shutdown():
    await http_client.aclose()
    log.info("HTTP client đã đóng.")

def image_to_base64_data_uri(file_path: Path) -> Optional[str]:
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        ext = file_path.suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        log.error(f"Lỗi khi chuyển ảnh sang base64: {e}")
        return None

async def check_and_rename_chat(session_id: int, user_prompt: str, ai_response: str, db: Session):
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session or session.title != "New Chat":
            return

        log.info(f"Đang tạo tự động tiêu đề cho Session ID: {session_id}")
        title_prompt = (
            f"Based on the following conversation start, create a very short, concise title (6 words or less, in English or Vietnamese depending on the context). "
            f"Do not add quotes or any prefix like 'Title:'. Just return the title itself.\n\n"
            f"User: \"{user_prompt}\"\n"
            f"Assistant: \"{ai_response}\"\n\n"
            f"Title:"
        )
        messages = [{"role": "user", "content": title_prompt}]
        
        response = await http_client.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={"messages": messages, "stream": False, "temperature": 0.2, "max_tokens": 20}
        )
        
        if response.status_code == 200:
            data = response.json()
            new_title = data['choices'][0]['message']['content'].strip().replace("\"", "")
            session.title = new_title
            db.commit()
            log.info(f"Session {session_id} đã được đổi tên thành: {new_title}")
        else:
            log.error(f"Lỗi khi gọi Llama Server để lấy title: {response.text}")
    except Exception as e:
        log.error(f"Lỗi trong tác vụ nền (rename chat): {e}", exc_info=True)
        db.rollback()

async def process_chat_request(
    db: Session,
    session_id: int,
    prompt: str,
    use_rag: bool,
    image: Optional[UploadFile] = None,
    emotion_snapshot: Optional[UploadFile] = None,
    emotion_history_json: str = "[]"
):
    image_path_str = None
    if image:
        try:
            ext = Path(image.filename).suffix if image.filename else ".jpg"
            file_path = TEMP_IMAGE_DIR / f"{session_id}_{datetime.now().timestamp()}{ext}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            image_path_str = str(file_path.relative_to(ROOT_DIR))
            log.info(f"Đã lưu ảnh đính kèm tại: {image_path_str}")
        except Exception as e:
            log.error(f"Lỗi lưu ảnh upload: {e}", exc_info=True)
            image_path_str = None
        finally:
            if image: image.file.close()

    db_user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=prompt,
        image_url=image_path_str
    )
    db.add(db_user_msg)
    
    db.commit() 
    db.refresh(db_user_msg)
    
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.now()
        db.commit()
    
    memory_block = get_user_memory_block(db)
    dynamic_system_prompt = f"{EMOTICA_SYSTEM_PROMPT}\n{memory_block}"
    
    llama_messages = []

    history_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
    
    for msg in history_msgs:
        content_parts = []
        if msg.content:
            content_parts.append({"type": "text", "text": msg.content})
        
        if msg.image_url:
            img_full_path = ROOT_DIR / msg.image_url
            if img_full_path.exists():
                b64_uri = image_to_base64_data_uri(img_full_path)
                if b64_uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": b64_uri}})
            else:
                log.warning(f"Không tìm thấy ảnh lịch sử: {img_full_path}")
        
        if msg.id == db_user_msg.id: 
            current_prompt_text = msg.content
            
            if emotion_snapshot:
                try:
                    snapshot_bytes = await emotion_snapshot.read()
                    emotion = emotion_service.get_emotion(snapshot_bytes)
                    if emotion not in ["no_face", "error", "unknown_result", "error_decoding_image"]:
                        log.info(f"Phát hiện cảm xúc (snapshot): {emotion}")
                        current_prompt_text = f"(User is feeling: {emotion})\n\n{current_prompt_text}"
                except Exception as e:
                    log.error(f"Lỗi xử lý emotion snapshot: {e}")
                finally:
                    if emotion_snapshot: emotion_snapshot.file.close()
            
            try:
                emotion_history = json.loads(emotion_history_json)
                if emotion_history:
                    history_str = ", ".join(emotion_history)
                    log.info(f"Phát hiện hành trình cảm xúc: {history_str}")
                    current_prompt_text = f"(User's emotional journey: {history_str})\n\n{current_prompt_text}"
            except Exception as e:
                log.error(f"Lỗi parse emotion_history_json: {e}")

            
            
            
            final_user_prompt = (
                f"{dynamic_system_prompt}\n\n"
                f"--- INSTRUCTIONS END, USER REQUEST BEGINS ---\n\n"
                f"{current_prompt_text}"
            )
            
            if content_parts and content_parts[0]["type"] == "text":
                content_parts[0]["text"] = final_user_prompt
            else:
                content_parts.insert(0, {"type": "text", "text": final_user_prompt})
        
        llama_messages.append({"role": msg.role, "content": content_parts})
        
    return db_user_msg, llama_messages

@app.get("/", response_class=RedirectResponse)
async def redirect_to_chat():
    return RedirectResponse(url="/chat")

@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def get_specific_chat_page(request: Request, session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        return RedirectResponse(url="/chat")
    return templates.TemplateResponse("chat.html", {"request": request, "session_id": session_id})

@app.get("/rag", response_class=HTMLResponse)
async def get_rag_page(request: Request):
    return templates.TemplateResponse("rag.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/memory", response_class=HTMLResponse)
async def get_memory_page(request: Request):
    return templates.TemplateResponse("memory.html", {"request": request})

@app.post("/api/chat/new", response_model=dict)
async def create_new_chat_session(db: Session = Depends(get_db)):
    try:
        new_session = ChatSession()
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        log.info(f"Tạo session mới, ID: {new_session.id}")
        return {"session_id": new_session.id, "title": new_session.title}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Không thể tạo session mới")

@app.get("/api/chat/history", response_model=List[dict])
async def get_chat_history(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]

@app.get("/api/chat/{session_id}/messages", response_model=List[dict])
async def get_chat_messages(session_id: int, db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
    result = []
    for m in messages:
        image_url = None
        if m.image_url:
            try:
                relative_path = Path(m.image_url).relative_to(TEMP_IMAGE_DIR.relative_to(ROOT_DIR))
                image_url = f"/temp_user_image/{relative_path}"
            except ValueError:
                image_url = f"/temp_user_image/{Path(m.image_url).name}"
        
        result.append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "image": image_url
        })
    return result

@app.delete("/api/chat/{session_id}")
async def delete_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Không tìm thấy session")
    db.delete(session)
    db.commit()
    return {"status": "success", "deleted_session_id": session_id}

@app.get("/api/tts")
async def text_to_speech(text: str):
    if not tts_analyzer:
        raise HTTPException(status_code=503, detail="Dịch vụ TTS chưa sẵn sàng")

    async def audio_stream_generator():
        try:
            audio_bytes = tts_analyzer.synthesize_speech(text, lang_code="auto")
            if audio_bytes:
                
                chunk_size = 8192 
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]
                    
                    await asyncio.sleep(0.001) 
            
        except Exception as e:
            log.error(f"Lỗi TTS streaming: {e}", exc_info=True)
            

    return StreamingResponse(audio_stream_generator(), media_type="audio/wav")

@app.post("/api/emotion/analyze")
async def analyze_emotion(file: UploadFile = File(...)):
    if not emotion_service:
        raise HTTPException(status_code=503, detail="Dịch vụ Emotion chưa sẵn sàng")
    image_bytes = await file.read()
    emotion = emotion_service.get_emotion(image_bytes)
    return {"emotion": emotion}

@app.post("/api/emotion/test")
async def test_emotion_analyzer(file: UploadFile = File(...)):
    if not emotion_service:
        raise HTTPException(status_code=503, detail="Dịch vụ Emotion chưa sẵn sàng")
    image_bytes = await file.read()
    drawn_image_bytes = emotion_service.test_draw_emotion(image_bytes)
    if drawn_image_bytes:
        return StreamingResponse(iter([drawn_image_bytes]), media_type="image/jpeg")
    else:
        raise HTTPException(status_code=500, detail="Không thể xử lý ảnh test")

@app.get("/api/system/stats")
async def proxy_system_stats():
    try:
        resp = await http_client.get(f"{LLAMA_SERVER_URL}/system/stats")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Không thể kết nối Llama Server (stats): {e}")

@app.get("/api/models/llama")
async def proxy_get_llama_models():
    try:
        resp = await http_client.get(f"{LLAMA_SERVER_URL}/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Không thể kết nối Llama Server (models): {e}")

@app.post("/api/models/llama/switch")
async def proxy_switch_llama_model(request: Request):
    try:
        body = await request.json()
        resp = await http_client.post(f"{LLAMA_SERVER_URL}/models/switch", json=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi khi chuyển model: {e}")

@app.get("/api/models/captioner")
async def proxy_get_captioner_models():
    try:
        resp = await http_client.get(f"{LLAMA_SERVER_URL}/captioner/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi: {e}")

@app.post("/api/models/captioner/switch")
async def proxy_switch_captioner_model(request: Request):
    try:
        body = await request.json()
        resp = await http_client.post(f"{LLAMA_SERVER_URL}/captioner/switch", json=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi: {e}")

@app.get("/api/rag/list")
async def proxy_rag_list():
    try:
        resp = await http_client.get(f"{LLAMA_SERVER_URL}/rag/list")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi: {e}")

@app.post("/api/rag/upload")
async def proxy_rag_upload(file: UploadFile = File(...)):
    try:
        files = {'file': (file.filename, await file.read(), file.content_type)}
        resp = await http_client.post(f"{LLAMA_SERVER_URL}/rag/upload", files=files)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi upload RAG: {e}")

@app.delete("/api/rag/delete")
async def proxy_rag_delete(filename: str):
    try:
        resp = await http_client.delete(f"{LLAMA_SERVER_URL}/rag/delete", params={"filename": filename})
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi xóa RAG: {e}")

@app.get("/api/models/config")
async def get_llama_config():
    if not LLAMA_CONFIG_FILE.exists():
        raise HTTPException(status_code=404, detail="models/cfg.json not found")
    try:
        with open(LLAMA_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return JSONResponse(content=config_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config file: {e}")

@app.post("/api/models/config")
async def save_llama_config(new_config: dict = Body(...)):
    try:
        if LLAMA_CONFIG_FILE.exists():
            backup_path = LLAMA_CONFIG_FILE.with_suffix(f".{datetime.now().timestamp()}.bak")
            shutil.copy(LLAMA_CONFIG_FILE, backup_path)
            log.info(f"Đã backup config cũ tại: {backup_path}")
            
        with open(LLAMA_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
            
        return {"status": "success", "message": "Config saved. Reloading Llama Server..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config file: {e}")

@app.post("/api/models/config/reload")
async def proxy_reload_llama_config():
    try:
        resp = await http_client.post(f"{LLAMA_SERVER_URL}/models/reload-config")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Lỗi khi reload Llama Server: {e}")

@app.get("/api/memory", response_model=List[dict])
async def get_all_memory(db: Session = Depends(get_db)):
    memories = db.query(UserMemory).order_by(UserMemory.created_at.desc()).all()
    return [{"id": m.id, "content": m.content, "created_at": m.created_at} for m in memories]

@app.get("/api/memory/status", response_model=dict)
async def get_memory_status(db: Session = Depends(get_db)):
    current_count = db.query(func.count(UserMemory.id)).scalar()
    limit = 100
    return {
        "count": current_count,
        "limit": limit,
        "is_full": current_count >= limit
    }

@app.post("/api/memory", response_model=dict)
async def add_memory_manual(memory: MemoryUpdate, db: Session = Depends(get_db)):
    current_count = db.query(func.count(UserMemory.id)).scalar()
    if current_count >= 100:
        raise HTTPException(status_code=400, detail="Đã đạt giới hạn 100 mục. Hãy xóa bớt trước khi thêm.")
        
    db_memory = UserMemory(content=memory.content)
    db.add(db_memory)
    db.commit()
    db.refresh(db_memory)
    return {"id": db_memory.id, "content": db_memory.content, "created_at": db_memory.created_at}

@app.put("/api/memory/{memory_id}", response_model=dict)
async def update_memory(memory_id: int, memory: MemoryUpdate, db: Session = Depends(get_db)):
    db_memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not db_memory:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục memory")
    
    db_memory.content = memory.content
    db.commit()
    db.refresh(db_memory)
    return {"id": db_memory.id, "content": db_memory.content, "created_at": db_memory.created_at}

@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: int, db: Session = Depends(get_db)):
    db_memory = db.query(UserMemory).filter(UserMemory.id == memory_id).first()
    if not db_memory:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục memory")
    
    db.delete(db_memory)
    db.commit()
    return {"status": "success", "deleted_id": memory_id}

async def stream_and_save_response(
    llama_payload: dict,
    session_id: int,
    user_prompt_for_title: str,
    background_tasks: BackgroundTasks
) -> AsyncGenerator[str, None]:
    
    full_response = ""
    buffer = ""
    tag_processed = False
    in_tag = False
    tag_content = ""
    
    save_tag_start_regex = re.compile(r"<saveUserInfo>", re.IGNORECASE)
    save_tag_end_regex = re.compile(r"</saveUserInfo>", re.IGNORECASE)

    try:
        async with http_client.stream(
            "POST", 
            f"{LLAMA_SERVER_URL}/v1/chat/completions", 
            json=llama_payload, 
            timeout=None
        ) as response:
            
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        
                        if in_tag:
                            log.warning("Stream kết thúc trong khi đang trong tag <saveUserInfo>")
                        yield "data: [DONE]\n\n"
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if "error" in chunk:
                            log.error(f"Lỗi từ Llama Server Stream: {chunk['error']}")
                            raise HTTPException(status_code=500, detail=chunk['error'])
                            
                        delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content')
                        
                        if delta:
                            if not tag_processed:
                                buffer += delta
                                
                                
                                if not in_tag:
                                    match_start = save_tag_start_regex.search(buffer)
                                    if match_start:
                                        in_tag = True
                                        tag_content = buffer[match_start.end():]
                                        buffer = buffer[:match_start.start()]
                                        
                                        if buffer.strip():
                                            full_response += buffer
                                            new_chunk_data = {
                                                "choices": [{"delta": {"content": buffer}}],
                                                "created": chunk.get("created"),
                                                "id": chunk.get("id"),
                                                "model": chunk.get("model"),
                                                "object": chunk.get("object")
                                            }
                                            yield f"data: {json.dumps(new_chunk_data)}\n\n"
                                        buffer = ""
                                
                                if in_tag:
                                    
                                    tag_content += delta
                                    
                                    match_end = save_tag_end_regex.search(tag_content)
                                    if match_end:
                                        memory_content = tag_content[:match_end.start()].strip()
                                        remaining_content = tag_content[match_end.end():].strip()
                                        
                                        log.info(f"Phát hiện tag <saveUserInfo>, nội dung: {memory_content}")
                                        background_tasks.add_task(save_user_memory_task, memory_content, SessionLocal())
                                        
                                        in_tag = False
                                        tag_processed = True
                                        tag_content = ""
                                        
                                        
                                        if remaining_content:
                                            full_response += remaining_content
                                            new_chunk_data = {
                                                "choices": [{"delta": {"content": remaining_content}}],
                                                "created": chunk.get("created"),
                                                "id": chunk.get("id"),
                                                "model": chunk.get("model"),
                                                "object": chunk.get("object")
                                            }
                                            yield f"data: {json.dumps(new_chunk_data)}\n\n"
                                    continue  
                                
                                
                                if buffer.strip() and not buffer.lstrip().startswith("<"):
                                    tag_processed = True
                                    full_response += buffer
                                    new_chunk_data = {
                                        "choices": [{"delta": {"content": buffer}}],
                                        "created": chunk.get("created"),
                                        "id": chunk.get("id"),
                                        "model": chunk.get("model"),
                                        "object": chunk.get("object")
                                    }
                                    yield f"data: {json.dumps(new_chunk_data)}\n\n"
                                    buffer = ""
                            else:
                                full_response += delta
                                new_chunk_data = {
                                    "choices": [{"delta": {"content": delta}}],
                                    "created": chunk.get("created"),
                                    "id": chunk.get("id"),
                                    "model": chunk.get("model"),
                                    "object": chunk.get("object")
                                }
                                yield f"data: {json.dumps(new_chunk_data)}\n\n"
                                
                    except json.JSONDecodeError:
                        continue
            
        if full_response:
            with SessionLocal() as db_session:
                db_ai_msg = ChatMessage(session_id=session_id, role="assistant", content=full_response)
                db_session.add(db_ai_msg)
                
                session = db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
                if session and session.title == "New Chat":
                    db_session.commit()
                    background_tasks.add_task(check_and_rename_chat, session_id, user_prompt_for_title, full_response, SessionLocal())
                else:
                    db_session.commit()
                    
    except Exception as e:
        log.error(f"Lỗi nghiêm trọng khi stream chat: {e}", exc_info=True)
        error_chunk = {"error": {"message": f"Internal Server Error: {e}"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

def lstrip_if_possible(s):
    try:
        return s.lstrip()
    except Exception:
        return s

@app.post("/api/chat/send")
async def send_message_stream(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    session_id: int = Form(...),
    prompt: str = Form(""),
    use_rag: bool = Form(False),
    model: str = Form(...),
    emotion_history_json: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
    emotion_snapshot: Optional[UploadFile] = File(None)
):
    
    db_user_msg, llama_messages = await process_chat_request(
        db, session_id, prompt, use_rag, image, emotion_snapshot, emotion_history_json
    )
    
    llama_payload = {
        "messages": llama_messages,
        "stream": True,
        "use_rag": use_rag
    }
    log.info(f"Gửi request (Stream) đến Llama Server (RAG: {use_rag})")
    
    return StreamingResponse(
        stream_and_save_response(
            llama_payload,
            session_id,
            prompt,
            background_tasks
        ),
        media_type="text/event-stream"
    )

@app.post("/api/chat/regenerate")
async def regenerate_message_stream(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    message_id: int = Form(...),
    new_content: str = Form(...),
    use_rag: bool = Form(False),
):
    log.info(f"Yêu cầu Regenerate cho Message ID: {message_id}")
    
    msg_to_edit = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    
    if not msg_to_edit:
        raise HTTPException(status_code=404, detail="Không tìm thấy tin nhắn")
    if msg_to_edit.role != 'user':
        raise HTTPException(status_code=400, detail="Chỉ có thể sửa tin nhắn của người dùng")
        
    session_id = msg_to_edit.session_id
    edit_timestamp = msg_to_edit.timestamp

    try:
        deleted_count = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.timestamp > edit_timestamp
        ).delete(synchronize_session=False)
        log.info(f"Đã xóa {deleted_count} tin nhắn sau tin nhắn được sửa.")
        
        msg_to_edit.content = new_content
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.now()
            
        db.commit()
        
    except Exception as e:
        db.rollback()
        log.error(f"Lỗi khi xóa/cập nhật DB để regenerate: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi DB: {e}")
        
    
    memory_block = get_user_memory_block(db)
    dynamic_system_prompt = f"{EMOTICA_SYSTEM_PROMPT}\n{memory_block}"
    
    llama_messages = []

    history_msgs = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp.asc()).all()
    
    for i, msg in enumerate(history_msgs):
        content_parts = []
        if msg.content:
            content_parts.append({"type": "text", "text": msg.content})
        
        if msg.image_url:
            img_full_path = ROOT_DIR / msg.image_url
            if img_full_path.exists():
                b64_uri = image_to_base64_data_uri(img_full_path)
                if b64_uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": b64_uri}})
        
        is_last_message = (i == len(history_msgs) - 1)
        if is_last_message and msg.role == 'user':
            prompt_text = msg.content
            
            
            
            
            final_user_prompt = (
                f"{dynamic_system_prompt}\n\n"
                f"--- INSTRUCTIONS END, USER REQUEST BEGINS ---\n\n"
                f"{prompt_text}"
            )
            
            if content_parts and content_parts[0]["type"] == "text":
                content_parts[0]["text"] = final_user_prompt
            else:
                content_parts.insert(0, {"type": "text", "text": final_user_prompt})

        llama_messages.append({"role": msg.role, "content": content_parts})
        
    llama_payload = {
        "messages": llama_messages,
        "stream": True,
        "use_rag": use_rag
    }
    log.info(f"Gửi request (Regenerate) đến Llama Server (RAG: {use_rag})")
    
    return StreamingResponse(
        stream_and_save_response(
            llama_payload,
            session_id,
            new_content,
            background_tasks
        ),
        media_type="text/event-stream"
    )

@app.post("/api/chat/send_non_stream")
async def send_message_non_stream(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    session_id: int = Form(...),
    prompt: str = Form(""),
    use_rag: bool = Form(False),
    model: str = Form(...), 
    emotion_history_json: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
    emotion_snapshot: Optional[UploadFile] = File(None)
):
    
    db_user_msg, llama_messages = await process_chat_request(
        db, session_id, prompt, use_rag, image, emotion_snapshot, emotion_history_json
    )
    
    llama_payload = {
        "messages": llama_messages,
        "stream": False,
        "use_rag": use_rag
    }
    log.info(f"Gửi request (Non-Stream) đến Llama Server (RAG: {use_rag})")
    
    try:
        response = await http_client.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json=llama_payload
        )
        response.raise_for_status()
        data = response.json()
        full_response = data['choices'][0]['message']['content'].strip()
        
        content_for_db = full_response
        
        save_tag_regex = re.compile(r"^\s*<saveUserInfo>(.*?)</saveUserInfo>", re.DOTALL | re.IGNORECASE)
        match = save_tag_regex.search(full_response)
        if match:
            memory_content = match.group(1).strip()
            remaining_content = full_response[match.end():].lstrip()
            
            log.info(f"(Non-Stream) Phát hiện tag <saveUserInfo>, nội dung: {memory_content}")
            background_tasks.add_task(save_user_memory_task, memory_content, SessionLocal())
            
            content_for_db = remaining_content
            full_response = remaining_content
            
        if content_for_db:
            db_ai_msg = ChatMessage(session_id=session_id, role="assistant", content=content_for_db)
            db.add(db_ai_msg)
            
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session and session.title == "New Chat":
                db.commit()
                background_tasks.add_task(check_and_rename_chat, session_id, prompt, full_response, SessionLocal())
            else:
                db.commit()
        
        user_image_url = None
        if db_user_msg.image_url:
            try:
                relative_path = Path(db_user_msg.image_url).relative_to(TEMP_IMAGE_DIR.relative_to(ROOT_DIR))
                user_image_url = f"/temp_user_image/{relative_path}"
            except ValueError:
                user_image_url = f"/temp_user_image/{Path(db_user_msg.image_url).name}"

        return JSONResponse(content={
            "role": "assistant",
            "content": full_response,
            "user_message": {
                "id": db_user_msg.id,
                "role": "user",
                "content": prompt,
                "image": user_image_url
            }
        })

    except Exception as e:
        log.error(f"Lỗi Non-Stream chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ: {e}")

if __name__ == "__main__":
    log.info(f"--- Khởi chạy Emotica AI Web Server trên http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT} ---")
    uvicorn.run("main:app", host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)