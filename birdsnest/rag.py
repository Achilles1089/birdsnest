########################################################################################################
# Bird's Nest — RAG Pipeline
# Document ingestion, embedding, vector storage (FAISS), and retrieval
########################################################################################################

import os, json, hashlib, time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class RAGPipeline:
    """Local RAG pipeline with document ingestion, embedding, and retrieval."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.docs_dir = os.path.join(data_dir, "documents")
        self.index_dir = os.path.join(data_dir, "faiss_index")
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        self._faiss_index = None
        self._embed_model = None
        self._metadata: List[Dict[str, Any]] = []  # parallel to FAISS vectors
        self._embed_dim = 384  # all-MiniLM-L6-v2 dimension

        # Load existing index if present
        self._load_index()

    # ── Persistence ───────────────────────────────────────────────────────

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "metadata.json")

    def _index_path(self) -> str:
        return os.path.join(self.index_dir, "vectors.faiss")

    def _load_index(self):
        """Load persisted FAISS index and metadata."""
        meta_path = self._meta_path()
        index_path = self._index_path()

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)

        if os.path.exists(index_path) and self._metadata:
            try:
                import faiss
                self._faiss_index = faiss.read_index(index_path)
            except Exception:
                self._faiss_index = None
                self._metadata = []

    def _save_index(self):
        """Persist FAISS index and metadata to disk."""
        if self._faiss_index is not None:
            import faiss
            faiss.write_index(self._faiss_index, self._index_path())

        with open(self._meta_path(), "w") as f:
            json.dump(self._metadata, f)

    def _ensure_faiss(self):
        """Create FAISS index if not loaded."""
        if self._faiss_index is not None:
            return
        import faiss
        self._faiss_index = faiss.IndexFlatIP(self._embed_dim)  # inner product (cosine on normalized)

    def _ensure_embedder(self):
        """Lazily load sentence-transformers embedding model."""
        if self._embed_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Document Ingestion ────────────────────────────────────────────────

    def ingest(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a document: extract text, chunk, embed, store in FAISS."""
        self._ensure_faiss()
        self._ensure_embedder()

        t0 = time.time()

        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        if filename is None:
            filename = os.path.basename(file_path)

        # Extract text
        text = self._extract_text(file_path)
        if not text.strip():
            return {"error": f"No text extracted from {filename}"}

        # Generate doc ID from content hash
        doc_id = hashlib.md5(f"{filename}:{text[:500]}".encode()).hexdigest()[:12]

        # Check if already ingested
        if any(m.get("doc_id") == doc_id for m in self._metadata):
            return {"status": "already_indexed", "doc_id": doc_id, "filename": filename}

        # Chunk the text
        chunks = self._chunk_text(text, chunk_size=512, overlap=64)
        if not chunks:
            return {"error": "No chunks generated"}

        # Embed all chunks — sentence-transformers returns L2-normalized vectors
        embeddings = self._embed_model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Add to FAISS
        self._faiss_index.add(embeddings)

        # Store metadata parallel to vectors
        for i, chunk in enumerate(chunks):
            self._metadata.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_idx": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })

        # Save source file
        dest = os.path.join(self.docs_dir, filename)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(file_path, dest)

        # Persist
        self._save_index()

        elapsed = round(time.time() - t0, 1)
        return {
            "status": "indexed",
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
            "time": elapsed,
        }

    # ── Retrieval ─────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        self._ensure_faiss()
        self._ensure_embedder()

        if self._faiss_index.ntotal == 0:
            return []

        query_vec = self._embed_model.encode([text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        k = min(top_k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                "text": meta["text"],
                "filename": meta.get("filename", "?"),
                "chunk_idx": meta.get("chunk_idx", 0),
                "score": round(float(scores[0][i]), 3),
            })

        return results

    def build_context(self, query_text: str, top_k: int = 5) -> str:
        """Build a context string from retrieved chunks for prompt injection."""
        chunks = self.query(query_text, top_k=top_k)
        if not chunks:
            return ""

        context_parts = []
        for c in chunks:
            context_parts.append(f"[Source: {c['filename']}]\n{c['text']}")

        return (
            "The following documents may be relevant to the user's question:\n\n"
            + "\n\n---\n\n".join(context_parts)
            + "\n\nUse the above context to inform your response. If the context is not relevant, ignore it.\n\n"
        )

    # ── Document Management ───────────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        docs: Dict[str, Dict] = {}
        for meta in self._metadata:
            doc_id = meta.get("doc_id", "?")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "?"),
                    "total_chunks": meta.get("total_chunks", 0),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1
        return list(docs.values())

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document and rebuild the FAISS index without it."""
        # Find chunks to keep vs remove
        keep_indices = []
        remove_count = 0
        filename = "?"

        for i, meta in enumerate(self._metadata):
            if meta.get("doc_id") == doc_id:
                remove_count += 1
                filename = meta.get("filename", "?")
            else:
                keep_indices.append(i)

        if remove_count == 0:
            return {"error": f"Document not found: {doc_id}"}

        # Rebuild index with remaining vectors
        import faiss
        if keep_indices and self._faiss_index.ntotal > 0:
            # Reconstruct vectors for kept indices
            kept_vectors = np.array(
                [self._faiss_index.reconstruct(i) for i in keep_indices],
                dtype=np.float32,
            )
            new_index = faiss.IndexFlatIP(self._embed_dim)
            new_index.add(kept_vectors)
            self._faiss_index = new_index
            self._metadata = [self._metadata[i] for i in keep_indices]
        else:
            self._faiss_index = faiss.IndexFlatIP(self._embed_dim)
            self._metadata = []

        # Remove source file
        doc_path = os.path.join(self.docs_dir, filename)
        if os.path.exists(doc_path):
            os.remove(doc_path)

        self._save_index()

        return {"deleted": doc_id, "filename": filename, "chunks_removed": remove_count}

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics."""
        docs = self.list_documents()
        total = self._faiss_index.ntotal if self._faiss_index else 0
        return {
            "total_documents": len(docs),
            "total_chunks": total,
            "documents": docs,
        }

    # ── Text Extraction ───────────────────────────────────────────────────

    def _extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        ext = Path(file_path).suffix.lower()

        if ext in ('.txt', '.md', '.py', '.js', '.ts', '.css', '.html', '.json', '.yaml', '.yml',
                    '.toml', '.cfg', '.ini', '.sh', '.bash', '.zsh', '.csv', '.log', '.rst'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.pdf':
            return self._extract_pdf(file_path)

        elif ext in ('.docx', '.doc'):
            return self._extract_docx(file_path)

        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                return ""

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            try:
                import fitz
                doc = fitz.open(file_path)
                return "".join(page.get_text() for page in doc)
            except ImportError:
                raise RuntimeError("No PDF library found. Run: pip install PyPDF2")

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    # ── Chunking ──────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    for sep in ['. ', '.\n', '! ', '? ', ';\n']:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks
