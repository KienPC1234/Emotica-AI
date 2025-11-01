import os
import gc
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.docstore.document import Document


from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class RAGManager:
    """
    Quản lý cơ sở dữ liệu vector (FAISS) cho RAG.
    Hỗ trợ: .pdf, .txt, .md
    Hỗ trợ đa ngôn ngữ (Việt, Anh) qua SentenceTransformers.
    """
    
    def __init__(self, 
                 db_data_dir: str = "vectordb/data", 
                 docs_dir: str = "vectordb/rawcontents",
                 embedding_model_dir: str = "embedding_models",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        
        
        self.base_path = Path(os.path.dirname(__file__)) 
        
        
        self.db_path = str(self.base_path / db_data_dir)
        self.docs_path = self.base_path / docs_dir
        self.metadata_file = self.base_path / db_data_dir / "metadata.json"
        
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.docs_path, exist_ok=True)
        
        log.info(f"RAG DB path (data): {self.db_path}")
        log.info(f"RAG Docs path (raw): {self.docs_path}")
        
        try:
            log.info(f"Đang tải embedding model: {embedding_model} (có thể mất chút thời gian)...")
            
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=embedding_model,
                cache_folder=str(self.base_path / embedding_model_dir),
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            log.critical(f"Không thể tải embedding model: {e}", exc_info=True)
            raise
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.documents_metadata = self._load_metadata()
        self.vector_store = self._load_or_create_db()
        log.info("RAG Manager đã sẵn sàng.")

    def _load_metadata(self) -> Dict[str, List[str]]:
        """Tải metadata (file -> [chunk_ids]) từ file JSON."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Lỗi đọc {self.metadata_file}, tạo mới. Lỗi: {e}")
        return {}

    def _save_metadata(self):
        """Lưu metadata (file -> [chunk_ids]) vào file JSON."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2)
        except Exception as e:
            log.error(f"Không thể lưu metadata: {e}", exc_info=True)

    def _load_or_create_db(self) -> FAISS:
        """Tải FAISS index từ disk nếu có, nếu không thì tạo mới."""
        index_file = os.path.join(self.db_path, "index.faiss")
        
        if os.path.exists(index_file):
            try:
                log.info("Đang tải FAISS index từ disk...")
                
                db = FAISS.load_local(
                    self.db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                log.info("Tải FAISS index thành công.")
                return db
            except Exception as e:
                log.warning(f"Lỗi tải FAISS index, tạo mới. Lỗi: {e}")
        
        log.info("Không tìm thấy index, tạo FAISS DB mới.")
        
        db = FAISS.from_texts(
            ["Bắt đầu cơ sở dữ liệu RAG"], 
            self.embeddings
        )
        db.save_local(self.db_path)
        return db

    def _load_and_split_document(self, file_path: Path) -> List[Document]:
        """Tùy theo loại file, sử dụng loader phù hợp và cắt nhỏ."""
        ext = file_path.suffix.lower()
        log.info(f"Đang tải file {file_path.name} (loại: {ext})")
        
        loader_map = {
            ".pdf": PyMuPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }
        
        if ext not in loader_map:
            raise ValueError(f"Loại file '{ext}' không được hỗ trợ.")
            
        loader_class = loader_map[ext]
        
        if ext == ".txt":
            loader = loader_class(str(file_path), encoding='utf-8')
        else:
            loader = loader_class(str(file_path))
            
        docs = loader.load()
        
        
        for doc in docs:
            doc.metadata["source_file"] = file_path.name
            
        chunks = self.text_splitter.split_documents(docs)
        log.info(f"Đã chia {file_path.name} thành {len(chunks)} chunks.")
        return chunks

    def add_document(self, temp_file_path: Path, filename: str) -> bool:
        """
        Thêm một tài liệu mới vào DB.
        1. Xóa (nếu đã tồn tại).
        2. Copy file vào thư mục quản lý.
        3. Tải và chia nhỏ (chunks).
        4. Thêm chunks vào FAISS.
        5. Cập nhật metadata.
        6. Lưu DB.
        """
        if filename in self.documents_metadata:
            log.info(f"File '{filename}' đã tồn tại. Đang xóa phiên bản cũ trước...")
            self.delete_document(filename)
        
        try:
            
            managed_file_path = self.docs_path / filename
            shutil.copy(str(temp_file_path), str(managed_file_path))
            
            
            chunks = self._load_and_split_document(managed_file_path)
            if not chunks:
                log.warning(f"File {filename} không có nội dung hoặc không thể xử lý.")
                managed_file_path.unlink() 
                return False

            
            chunk_ids = self.vector_store.add_documents(chunks, return_ids=True)
            
            
            self.documents_metadata[filename] = chunk_ids
            
            
            self.vector_store.save_local(self.db_path)
            self._save_metadata()
            
            log.info(f"Thêm thành công '{filename}' ({len(chunks)} chunks) vào RAG DB.")
            return True
            
        except Exception as e:
            log.error(f"Lỗi khi thêm tài liệu {filename}: {e}", exc_info=True)
            
            if 'managed_file_path' in locals() and managed_file_path.exists():
                managed_file_path.unlink()
            return False
        finally:
            
            if temp_file_path.exists() and "temp" in str(temp_file_path):
                temp_file_path.unlink()

    def delete_document(self, filename: str) -> bool:
        """Xóa một tài liệu khỏi DB (cả file và các chunks trong vector store)."""
        if filename not in self.documents_metadata:
            log.warning(f"Không tìm thấy file '{filename}' trong RAG DB.")
            return False
        
        try:
            
            ids_to_delete = self.documents_metadata[filename]
            
            
            if ids_to_delete:
                deleted = self.vector_store.delete(ids_to_delete)
                if not deleted:
                    log.warning(f"FAISS báo cáo không thể xóa IDs: {ids_to_delete}")
            
            
            del self.documents_metadata[filename]
            
            
            managed_file_path = self.docs_path / filename
            if managed_file_path.exists():
                managed_file_path.unlink()
                
            
            self.vector_store.save_local(self.db_path)
            self._save_metadata()
            
            log.info(f"Đã xóa thành công '{filename}' khỏi RAG DB.")
            return True
            
        except Exception as e:
            log.error(f"Lỗi khi xóa tài liệu {filename}: {e}", exc_info=True)
            return False

    def list_documents(self) -> List[str]:
        """Trả về danh sách các file đang được quản lý."""
        return list(self.documents_metadata.keys())

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Tìm kiếm các chunks liên quan nhất đến câu hỏi."""
        if not self.vector_store:
            log.error("RAG DB chưa được tải. Không thể tìm kiếm.")
            return []
            
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            
            formatted_results = [
                {
                    "source": doc.metadata.get("source_file", "unknown"),
                    "content": doc.page_content
                }
                for doc in results
            ]
            return formatted_results
        except Exception as e:
            log.error(f"Lỗi khi tìm kiếm RAG: {e}", exc_info=True)
            return []


try:
    
    rag_manager = RAGManager(
        db_data_dir="vectordb/data",
        docs_dir="vectordb/rawcontents",
        embedding_model_dir="embedding_models"
    )
except Exception as e:
    log.critical(f"Không thể khởi tạo RAGManager. Lỗi nghiêm trọng: {e}", exc_info=True)
    rag_manager = None


if __name__ == "__main__":
    if rag_manager:
        log.info("--- Bắt đầu Test Module RAGManager ---")
        
        
        test_txt_path = Path("test_rag_vi.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f:
            f.write("Đây là tài liệu tiếng Việt. Hà Nội là thủ đô của Việt Nam.")
            
        test_md_path = Path("test_rag_en.md")
        with open(test_md_path, "w", encoding="utf-8") as f:
            f.write("Hello\nThis is a *Markdown* file about the Eiffel Tower in Paris.")
        
        
        log.info("--- Test: Thêm tài liệu ---")
        rag_manager.add_document(test_txt_path, "tieng_viet.txt")
        rag_manager.add_document(test_md_path, "tieng_anh.md")
        
        
        log.info("--- Test: Liệt kê tài liệu ---")
        docs = rag_manager.list_documents()
        log.info(f"Các file trong DB: {docs}")
        
        
        log.info("--- Test: Tìm kiếm (Việt) ---")
        results_vi = rag_manager.search("Thủ đô Việt Nam là gì?")
        log.info(f"Kết quả (Việt): {results_vi}")

        log.info("--- Test: Tìm kiếm (Anh) ---")
        results_en = rag_manager.search("Where is the Eiffel Tower?")
        log.info(f"Kết quả (Anh): {results_en}")
        
        
        log.info("--- Test: Xóa tài liệu ---")
        rag_manager.delete_document("tieng_viet.txt")
        docs_after_delete = rag_manager.list_documents()
        log.info(f"Các file còn lại: {docs_after_delete}")
        
        
        test_txt_path.unlink()
        test_md_path.unlink()
        rag_manager.delete_document("tieng_anh.md")
        
        log.info("--- Test Module Hoàn Tất ---")
    else:
        log.error("Khởi tạo RAGManager thất bại. Không thể chạy test.")

