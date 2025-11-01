import os
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
# Giảm log rác từ sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class RAGManager:
    """
    Quản lý cơ sở dữ liệu vector (FAISS) cho RAG.
    Hỗ trợ: .pdf, .txt, .md
    Hỗ trợ đa ngôn ngữ (Việt, Anh) qua SentenceTransformers.
    
    CẤU TRÚC THƯ MỤC:
    - (base_path)/
        - embedding_models/ (Nơi cache model embedding)
        - vectordb/
            - data/ (Nơi chứa index FAISS và metadata.json)
                - index.faiss
                - index.pkl
                - metadata.json
            - rawcontents/ (Nơi lưu trữ các file .pdf, .txt... gốc)
                - doc1.pdf
                - notes.txt
    """
    
    def __init__(self, 
                 db_data_dir: str = "vectordb/data", 
                 docs_dir: str = "vectordb/rawcontents",
                 embedding_model_dir: str = "embedding_models",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        
        # Xác định đường dẫn cơ sở
        self.base_path = Path(__file__).resolve().parent
        
        # Định nghĩa các đường dẫn quan trọng
        self.db_path_str = str(self.base_path / db_data_dir)
        self.docs_path = self.base_path / docs_dir
        self.metadata_file = self.base_path / db_data_dir / "metadata.json"
        self.embedding_cache_path = str(self.base_path / embedding_model_dir)
        
        os.makedirs(self.db_path_str, exist_ok=True)
        os.makedirs(self.docs_path, exist_ok=True)
        
        log.info(f"RAG DB path (data): {self.db_path_str}")
        log.info(f"RAG Docs path (raw): {self.docs_path}")
        
        try:
            log.info(f"Đang tải embedding model: {embedding_model} (cache: {self.embedding_cache_path})...")
            # Sử dụng 'cpu' là một lựa chọn tốt để không cạnh tranh VRAM với LLaMA.
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=embedding_model,
                cache_folder=self.embedding_cache_path,
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
        
        self.documents_metadata: Dict[str, List[str]] = self._load_metadata()
        self.vector_store: Optional[FAISS] = self._load_or_create_db()
        log.info("RAG Manager đã sẵn sàng.")

    def _load_metadata(self) -> Dict[str, List[str]]:
        """(Internal) Tải metadata (file -> [chunk_ids]) từ file JSON."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                log.warning(f"Lỗi đọc {self.metadata_file} (JSON rỗng hoặc lỗi). Tạo mới.")
                return {}
            except Exception as e:
                log.warning(f"Lỗi đọc {self.metadata_file}, tạo mới. Lỗi: {e}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """(Internal) Lưu metadata (file -> [chunk_ids]) vào file JSON."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Không thể lưu metadata: {e}", exc_info=True)

    def _load_or_create_db(self) -> Optional[FAISS]:
        """(Internal) Tải FAISS index từ disk nếu có, nếu không thì tạo mới."""
        index_file = Path(self.db_path_str) / "index.faiss"
        
        if index_file.exists():
            try:
                log.info("Đang tải FAISS index từ disk...")
                db = FAISS.load_local(
                    self.db_path_str, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                log.info("Tải FAISS index thành công.")
                return db
            except Exception as e:
                log.warning(f"Lỗi tải FAISS index, tạo mới. Lỗi: {e}")
        
        log.info("Không tìm thấy index, tạo FAISS DB mới (rỗng).")
        try:
            # Tạo một DB trống rỗng (cần một entry giả để khởi tạo)
            db = FAISS.from_texts(
                ["Khởi tạo cơ sở dữ liệu RAG"], 
                self.embeddings
            )
            # Xóa entry giả ngay lập tức
            ids_to_delete = list(db.index_to_docstore_id.values())
            db.delete(ids_to_delete)
            db.save_local(self.db_path_str)
            return db
        except Exception as e:
            log.error(f"Không thể tạo DB FAISS mới: {e}", exc_info=True)
            return None

    def _load_and_split_document(self, file_path: Path) -> List[Document]:
        """(Internal) Tùy theo loại file, sử dụng loader phù hợp và cắt nhỏ."""
        ext = file_path.suffix.lower()
        log.info(f"Đang tải file {file_path.name} (loại: {ext})")
        
        loader_map = {
            ".pdf": PyMuPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }
        
        loader_class = loader_map.get(ext)
        if not loader_class:
            raise ValueError(f"Loại file '{ext}' không được hỗ trợ.")
            
        try:
            if ext == ".txt":
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            docs = loader.load()
            
            # Gán metadata (rất quan trọng)
            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                
            chunks = self.text_splitter.split_documents(docs)
            log.info(f"Đã chia {file_path.name} thành {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            log.error(f"Lỗi khi tải/chia file {file_path.name}: {e}", exc_info=True)
            return [] # Trả về list rỗng nếu lỗi

    def add_document(self, temp_file_path: Path, filename: str) -> bool:
        """
        Thêm một tài liệu mới vào DB (logic thay thế nếu đã tồn tại).
        1. Xóa (nếu đã tồn tại).
        2. Copy file vào thư mục quản lý 'rawcontents'.
        3. Tải và chia nhỏ (chunks).
        4. Thêm chunks vào FAISS.
        5. Cập nhật metadata.
        6. Lưu DB.
        """
        if not self.vector_store:
            log.error("Vector store chưa được khởi tạo. Không thể thêm tài liệu.")
            return False

        if filename in self.documents_metadata:
            log.info(f"File '{filename}' đã tồn tại. Đang xóa phiên bản cũ trước...")
            self.delete_document(filename)
        
        managed_file_path = self.docs_path / filename
        try:
            # 1. Copy file vào 'rawcontents'
            shutil.copy(str(temp_file_path), str(managed_file_path))
            
            # 2. Tải và chia nhỏ
            chunks = self._load_and_split_document(managed_file_path)
            if not chunks:
                log.warning(f"File {filename} không có nội dung hoặc không thể xử lý.")
                managed_file_path.unlink() # Xóa file rác
                return False

            # 3. Thêm chunks vào FAISS và lấy IDs
            chunk_ids = self.vector_store.add_documents(chunks, return_ids=True)
            
            # 4. Cập nhật metadata
            self.documents_metadata[filename] = chunk_ids
            
            # 5. Lưu trạng thái
            self.vector_store.save_local(self.db_path_str)
            self._save_metadata()
            
            log.info(f"Thêm thành công '{filename}' ({len(chunks)} chunks) vào RAG DB.")
            return True
            
        except Exception as e:
            log.error(f"Lỗi khi thêm tài liệu {filename}: {e}", exc_info=True)
            # Dọn dẹp nếu có lỗi
            if managed_file_path.exists():
                managed_file_path.unlink()
            return False
        finally:
            # Xóa file temp (nếu nó là file temp thật)
            if temp_file_path.exists() and "temp" in str(temp_file_path.parent):
                temp_file_path.unlink()

    def delete_document(self, filename: str) -> bool:
        """Xóa một tài liệu khỏi DB (cả file và các chunks trong vector store)."""
        if not self.vector_store:
            log.error("Vector store chưa được khởi tạo. Không thể xóa.")
            return False

        if filename not in self.documents_metadata:
            log.warning(f"Không tìm thấy file '{filename}' trong RAG DB.")
            # Xóa file (nếu có) trong rawcontents
            (self.docs_path / filename).unlink(missing_ok=True)
            return False
        
        try:
            # 1. Lấy IDs của các chunks cần xóa
            ids_to_delete = self.documents_metadata[filename]
            
            # 2. Xóa khỏi FAISS
            if ids_to_delete:
                deleted = self.vector_store.delete(ids_to_delete)
                if not deleted:
                    log.warning(f"FAISS báo cáo không thể xóa IDs: {ids_to_delete}")
            
            # 3. Xóa khỏi metadata
            del self.documents_metadata[filename]
            
            # 4. Xóa file gốc
            managed_file_path = self.docs_path / filename
            managed_file_path.unlink(missing_ok=True)
                
            # 5. Lưu trạng thái
            self.vector_store.save_local(self.db_path_str)
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
            results: List[Document] = self.vector_store.similarity_search(query, k=k)
            
            # Format output
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

    def unload_db_sync(self) -> None:
        """(Internal) Giải phóng vector store khỏi bộ nhớ."""
        log.info("Đang giải phóng RAG vector store...")
        if self.vector_store:
            del self.vector_store
            self.vector_store = None
        self.documents_metadata = {}
        gc.collect()

    def _rebuild_metadata_from_store(self) -> Dict[str, List[str]]:
        """
        (Internal) [TỐI ƯU] Tạo lại file metadata từ một FAISS index đã build.
        Rất quan trọng cho hàm rebuild_database.
        """
        if not self.vector_store:
            log.error("Không có vector store để rebuild metadata.")
            return {}

        log.info("Đang xây dựng lại map metadata (docstore_id -> filename)...")
        metadata_map: Dict[str, List[str]] = {}
        
        # docstore chứa map: docstore_id -> Document
        docstore = self.vector_store.docstore
        # index_to_docstore_id chứa map: faiss_index (int) -> docstore_id (str)
        index_to_id = self.vector_store.index_to_docstore_id
        
        # Không cần faiss_index, chỉ cần docstore_id
        all_docstore_ids: Set[str] = set(index_to_id.values())

        for docstore_id in all_docstore_ids:
            doc: Optional[Document] = docstore.search(docstore_id)
            if doc:
                filename = doc.metadata.get("source_file")
                if filename:
                    if filename not in metadata_map:
                        metadata_map[filename] = []
                    metadata_map[filename].append(docstore_id)
        
        log.info(f"Đã map {len(metadata_map)} file từ docstore.")
        return metadata_map

    def rebuild_database(self) -> None:
        """
        [TÍNH NĂNG MỚI] Xây dựng lại toàn bộ DB từ thư mục 'rawcontents'.
        Rất nhanh vì chỉ I/O disk 1 lần.
        """
        log.info("--- [BẮT ĐẦU REBUILD RAG DATABASE] ---")
        
        # 1. Giải phóng DB cũ
        self.unload_db_sync()
        
        # 2. Xóa data cũ (index.faiss, index.pkl, metadata.json)
        try:
            if os.path.exists(self.db_path_str):
                shutil.rmtree(self.db_path_str)
            os.makedirs(self.db_path_str)
            log.info("Đã xóa RAG data cũ.")
        except Exception as e:
            log.error(f"Lỗi khi xóa RAG data cũ: {e}", exc_info=True)
            return

        # 3. Quét thư mục 'rawcontents'
        all_files = list(self.docs_path.glob('*.*')) # Hỗ trợ các ext đã biết
        supported_exts = {".pdf", ".txt", ".md"}
        
        if not all_files:
            log.warning("Không tìm thấy file nào trong 'rawcontents'. Tạo DB rỗng.")
            self._load_or_create_db() # Tạo DB rỗng
            return

        # 4. Tải và chia nhỏ TẤT CẢ file
        all_chunks: List[Document] = []
        for file_path in all_files:
            if file_path.suffix.lower() not in supported_exts:
                log.warning(f"Bỏ qua file không hỗ trợ: {file_path.name}")
                continue
            try:
                chunks = self._load_and_split_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                log.error(f"Lỗi khi xử lý file {file_path.name}: {e}")

        if not all_chunks:
            log.error("Không thể tạo chunks từ bất kỳ file nào. Tạo DB rỗng.")
            self._load_or_create_db() # Tạo DB rỗng
            return

        # 5. [TỐI ƯU] Tạo FAISS index MỘT LẦN DUY NHẤT
        log.info(f"Đang thêm {len(all_chunks)} chunks vào FAISS index mới...")
        self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        
        # 6. Lưu DB mới
        log.info("Đang lưu FAISS index mới xuống disk...")
        self.vector_store.save_local(self.db_path_str)
        
        # 7. Tạo lại metadata từ DB vừa build
        log.info("Đang tạo lại metadata.json...")
        self.documents_metadata = self._rebuild_metadata_from_store()
        self._save_metadata()
        
        log.info("--- [REBUILD RAG DATABASE HOÀN TẤT] ---")


# --- Khởi tạo Singleton ---
try:
    # Khởi tạo RAGManager khi import module này
    rag_manager = RAGManager(
        db_data_dir="vectordb/data",
        docs_dir="vectordb/rawcontents",
        embedding_model_dir="embedding_models"
    )
except Exception as e:
    log.critical(f"Không thể khởi tạo RAGManager. Lỗi nghiêm trọng: {e}", exc_info=True)
    rag_manager = None


# --- Test Script ---
if __name__ == "__main__":
    if rag_manager:
        log.info("--- Bắt đầu Test Module RAGManager ---")
        
        # Chuẩn bị file test
        test_txt_path = Path("test_rag_vi.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f:
            f.write("Đây là tài liệu tiếng Việt. Hà Nội là thủ đô của Việt Nam.")
            
        test_md_path = Path("test_rag_en.md")
        with open(test_md_path, "w", encoding="utf-8") as f:
            f.write("Hello\nThis is a *Markdown* file about the Eiffel Tower in Paris.")
        
        # --- Test: Thêm tài liệu ---
        log.info("--- Test: Thêm tài liệu ---")
        rag_manager.add_document(test_txt_path, "tieng_viet.txt")
        rag_manager.add_document(test_md_path, "tieng_anh.md")
        
        # --- Test: Liệt kê tài liệu ---
        log.info("--- Test: Liệt kê tài liệu ---")
        docs = rag_manager.list_documents()
        log.info(f"Các file trong DB: {docs}")
        assert "tieng_viet.txt" in docs
        assert "tieng_anh.md" in docs

        # --- Test: Tìm kiếm (Việt) ---
        log.info("--- Test: Tìm kiếm (Việt) ---")
        results_vi = rag_manager.search("Thủ đô Việt Nam là gì?")
        log.info(f"Kết quả (Việt): {results_vi}")
        assert "Hà Nội" in results_vi[0]["content"]

        # --- Test: Tìm kiếm (Anh) ---
        log.info("--- Test: Tìm kiếm (Anh) ---")
        results_en = rag_manager.search("Where is the Eiffel Tower?")
        log.info(f"Kết quả (Anh): {results_en}")
        assert "Paris" in results_en[0]["content"]
        
        # --- Test: Rebuild (MỚI) ---
        log.info("--- Test: Rebuild Database ---")
        rag_manager.rebuild_database()
        
        log.info("--- Test: Liệt kê sau khi Rebuild ---")
        docs_rebuilt = rag_manager.list_documents()
        log.info(f"Các file trong DB (sau rebuild): {docs_rebuilt}")
        assert "tieng_viet.txt" in docs_rebuilt
        assert "tieng_anh.md" in docs_rebuilt

        log.info("--- Test: Tìm kiếm sau khi Rebuild ---")
        results_rebuilt = rag_manager.search("Thủ đô Việt Nam là gì?")
        log.info(f"Kết quả (sau rebuild): {results_rebuilt}")
        assert "Hà Nội" in results_rebuilt[0]["content"]
        
        # --- Test: Xóa tài liệu ---
        log.info("--- Test: Xóa tài liệu ---")
        rag_manager.delete_document("tieng_viet.txt")
        docs_after_delete = rag_manager.list_documents()
        log.info(f"Các file còn lại: {docs_after_delete}")
        assert "tieng_viet.txt" not in docs_after_delete
        assert "tieng_anh.md" in docs_after_delete
        
        # --- Dọn dẹp ---
        test_txt_path.unlink()
        test_md_path.unlink()
        rag_manager.delete_document("tieng_anh.md")
        
        log.info("--- Test Module Hoàn Tất ---")
    else:
        log.error("Khởi tạo RAGManager thất bại. Không thể chạy test.")
