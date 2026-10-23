"""
rag_core.py
===========
Framework-agnostic core of Teleoptimal: PDF loading, FAISS index (cached on
disk), chunk selection, prompt building, Claude call and PDF rendering.
The Streamlit UI lives in app.py.

Configuration is read from Streamlit secrets first (Streamlit Cloud), then
from environment variables / .env (local runs).
"""

import os
import re
import json
import string
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF — faster + usually more accurate than pypdf
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

# ── configuration ──────────────────────────────────────────────────────────────

load_dotenv()


def cfg(name: str, default: str = "") -> str:
    """st.secrets first (Streamlit Cloud), then environment / .env."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


ANTHROPIC_API_KEY     = cfg("ANTHROPIC_API_KEY")
OPENAI_API_KEY        = cfg("OPENAI_API_KEY")
MAX_DAILY_MESSAGES    = int(cfg("MAX_DAILY_MESSAGES", "5"))       # per browser session
GLOBAL_DAILY_MESSAGES = int(cfg("GLOBAL_DAILY_MESSAGES", "200"))  # whole app, 24h window
SIMILARITY_THRESHOLD  = float(cfg("SIMILARITY_THRESHOLD", "0.20"))
PER_SOURCE_MIN        = int(cfg("PER_SOURCE_MIN", "2"))
TOP_K                 = int(cfg("TOP_K", "24"))
PDF_DIRECTORY         = cfg("PDF_DIRECTORY", "./pdfs")
MAX_CONTEXT_TOKENS    = int(cfg("MAX_CONTEXT_TOKENS", "20000"))
CLAUDE_MODEL          = cfg("CLAUDE_MODEL", "claude-sonnet-5-5")
# Comma-separated list. With more than one, the UI shows a model picker (handy for A/B tests).
MODEL_CHOICES         = [m.strip() for m in cfg("MODEL_CHOICES", CLAUDE_MODEL).split(",") if m.strip()]
TEMPERATURE           = cfg("TEMPERATURE", "0.1")                 # set to "" to omit the parameter
EMBEDDING_MODEL       = cfg("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE            = int(cfg("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP         = int(cfg("CHUNK_OVERLAP", "300"))

MAX_CANDIDATES = 500
CACHE_VERSION = 4  # 4: manifest uses content hashes (survives git clone / redeploys)

VECTOR_CACHE_DIR     = os.path.join(PDF_DIRECTORY, ".vector_cache")
VECTOR_INDEX_PATH    = os.path.join(VECTOR_CACHE_DIR, "faiss_index")
VECTOR_MANIFEST_PATH = os.path.join(VECTOR_CACHE_DIR, "manifest.json")

_CHARS_PER_TOKEN = 3.5
_FOLLOWUP_MAX_WORDS = 8

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("teleoptimal")

# ── helpers ────────────────────────────────────────────────────────────────────

def sanitize_filename(text: str, max_len: int = 50) -> str:
    valid = f"-_. {string.ascii_letters}{string.digits}"
    name = "".join(c if c in valid else "_" for c in text)
    name = re.sub(r"_+", "_", name).strip("_. ")
    name = name[:max_len].rstrip("_. ")
    return (name or f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}") + ".pdf"


def split_filename(raw: str, user_message: str) -> Tuple[str, str]:
    """Pull the [FILENAME: ...] tag out of Claude's reply -> (clean_text, filename)."""
    m = re.search(r"\[FILENAME:\s*([^\]]+)\]", raw)
    if not m:
        return raw, sanitize_filename(user_message[:60])
    text = re.sub(r"\[FILENAME:\s*[^\]]+\]", "", raw).strip()
    name = m.group(1).strip().replace(" ", "_")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    valid = f"-_.{string.ascii_letters}{string.digits}"
    name = "".join(c for c in name if c in valid)
    if len(name) > 50:
        name = name[:46] + ".pdf"
    if len(name) < 5:
        name = f"respuesta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return text, name


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(text))


def _paragraph(text: str, style) -> Paragraph:
    try:
        return Paragraph(_inline_md(text), style)
    except Exception:
        return Paragraph(_esc(text), style)


def build_response_pdf(user_message: str, ai_response: str, sources: Optional[List[str]] = None) -> BytesIO:
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    justify = ParagraphStyle("Justify", parent=styles["Normal"], alignment=TA_JUSTIFY,
                             fontSize=11, leading=15, spaceAfter=8)
    code_style = ParagraphStyle("Code", parent=styles["Normal"], fontName="Courier",
                                fontSize=9, leading=12, leftIndent=12, spaceAfter=8,
                                backColor="#F4F4F4")
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm)
    content = []

    content += [Paragraph("Query", styles["Heading2"]), HRFlowable(width="100%", thickness=0.5),
                Spacer(1, 0.2 * cm), Paragraph(_esc(user_message), justify), Spacer(1, 0.4 * cm)]
    content += [Paragraph("Response", styles["Heading2"]), HRFlowable(width="100%", thickness=0.5),
                Spacer(1, 0.2 * cm)]

    in_code, code_lines = False, []
    for line in ai_response.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                if code_lines:
                    content.append(Paragraph("<br/>".join(_esc(l) for l in code_lines), code_style))
                code_lines, in_code = [], False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
        elif line.startswith("### "):
            content.append(_paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("## "):
            content.append(_paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("# "):
            content.append(_paragraph(line[2:], styles["Heading1"]))
        elif line.strip():
            content.append(_paragraph(line, justify))
        else:
            content.append(Spacer(1, 0.2 * cm))
    if in_code and code_lines:
        content.append(Paragraph("<br/>".join(_esc(l) for l in code_lines), code_style))

    if sources:
        content += [Spacer(1, 0.6 * cm), Paragraph("Textos consultados", styles["Heading2"]),
                    HRFlowable(width="100%", thickness=0.5), Spacer(1, 0.2 * cm)]
        for i, src in enumerate(sources, 1):
            content.append(Paragraph(f"{i}. {_esc(src)}", styles["Normal"]))
        content.append(Spacer(1, 0.4 * cm))

    content += [Spacer(1, 0.2 * cm), HRFlowable(width="100%", thickness=0.3),
                Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])]
    doc.build(content)
    buffer.seek(0)
    return buffer


# ── rate limiting ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding 24h window, thread-safe (Streamlit runs each session in its own thread)."""

    def __init__(self, max_messages: int):
        self.max_messages = max_messages
        self._hits: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def _prune(self, key: str) -> None:
        cutoff = datetime.now() - timedelta(days=1)
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]

    def can_send(self, key: str = "all") -> bool:
        with self._lock:
            self._prune(key)
            return len(self._hits[key]) < self.max_messages

    def record(self, key: str = "all") -> None:
        with self._lock:
            self._hits[key].append(datetime.now())

    def remaining(self, key: str = "all") -> int:
        with self._lock:
            self._prune(key)
            return max(0, self.max_messages - len(self._hits[key]))


# ── chunk selection ────────────────────────────────────────────────────────────

def select_chunks(scored, threshold, per_source_min, top_k, token_budget, chars_per_token):
    """
    scored: list of (doc, cosine) sorted by cosine DESCENDING.
    1. Floor: the `per_source_min` best chunks of every source are candidates.
    2. Fill: best remaining chunks >= threshold, up to `top_k` candidates.
    3. Budget: walk best-first, skipping chunks that no longer fit.
    4. Present: grouped by source (best source first), reading order inside each.
    """
    chosen = set()
    per_source = defaultdict(int)

    if per_source_min > 0:
        for idx, (doc, _s) in enumerate(scored):
            src = doc.metadata.get("source", "?")
            if per_source[src] < per_source_min:
                per_source[src] += 1
                chosen.add(idx)

    for idx, (_doc, s) in enumerate(scored):
        if len(chosen) >= top_k:
            break
        if idx not in chosen and s >= threshold:
            chosen.add(idx)

    kept, budget = [], token_budget
    for idx in sorted(chosen):
        doc, s = scored[idx]
        tokens = len(doc.page_content) / chars_per_token
        if tokens > budget:
            continue
        budget -= tokens
        kept.append((doc, s))

    best = {}
    for doc, s in kept:
        src = doc.metadata.get("source", "?")
        best[src] = max(best.get(src, -1.0), s)
    kept.sort(key=lambda t: (
        -best[t[0].metadata.get("source", "?")],
        t[0].metadata.get("source", "?"),
        t[0].metadata.get("start_index", 0),
    ))
    return kept


# ── document store ─────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DocumentStore:
    """PDFs -> chunks -> OpenAI embeddings -> FAISS (cached on disk)."""

    def __init__(self, pdf_directory: str = PDF_DIRECTORY):
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.pdf_files: List[str] = []
        self.indexed_files: List[str] = []
        self.empty_files: List[str] = []
        self._lock = threading.Lock()
        self._load()

    # ── PDF reading ──

    def _read_pdf(self, path: str) -> Tuple[str, int, int]:
        try:
            if _HAS_PYMUPDF:
                with fitz.open(path) as doc:
                    pages = [page.get_text() for page in doc]
            else:
                pages = [p.extract_text() or "" for p in PdfReader(path).pages]
            text = "\n".join(pages)
            n_text = sum(1 for p in pages if len(p.strip()) >= 50)
            logger.info("Read PDF %s (%d pages, %d with text, %d chars)", path, len(pages), n_text, len(text))
            return text, len(pages), n_text
        except Exception as exc:
            logger.error("Error reading %s: %s", path, exc)
            return "", 0, 0

    def _scan_pdfs(self) -> List[str]:
        os.makedirs(self.pdf_directory, exist_ok=True)
        return sorted(f for f in os.listdir(self.pdf_directory) if f.lower().endswith(".pdf"))

    # ── cache ──

    @staticmethod
    def _config_fingerprint() -> dict:
        return {
            "cache_version": CACHE_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "extractor": "pymupdf" if _HAS_PYMUPDF else "pypdf",
        }

    def _build_manifest(self, pdf_files: List[str]) -> dict:
        # Content hash instead of mtime: a fresh `git clone` on the host resets
        # mtimes, which would invalidate a committed index on every deploy.
        files = {}
        for f in pdf_files:
            path = os.path.join(self.pdf_directory, f)
            files[f] = [os.path.getsize(path), _sha256(path)]
        return {"config": self._config_fingerprint(), "files": files}

    def _try_load_cache(self, manifest: dict):
        if not os.path.isdir(VECTOR_INDEX_PATH) or not os.path.exists(VECTOR_MANIFEST_PATH):
            return None
        try:
            with open(VECTOR_MANIFEST_PATH) as fh:
                cached = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if cached.get("config") != manifest["config"]:
            logger.info("Cache invalid: indexing configuration changed.")
            return None
        if cached.get("files") != manifest["files"]:
            logger.info("Cache invalid: PDF set changed.")
            return None
        try:
            # The index is our own file (never user-supplied), so unpickling is safe here.
            store = FAISS.load_local(VECTOR_INDEX_PATH, self.embeddings,
                                     allow_dangerous_deserialization=True)
        except Exception as exc:
            logger.warning("Cache load failed, will rebuild: %s", exc)
            return None
        logger.info("Index restored from cache (embedding step skipped).")
        return store, cached.get("indexed", []), cached.get("empty", [])

    def _save_cache(self, store, manifest: dict, indexed: List[str], empty: List[str]) -> None:
        try:
            os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
            store.save_local(VECTOR_INDEX_PATH)
            with open(VECTOR_MANIFEST_PATH, "w") as fh:
                json.dump({**manifest, "indexed": indexed, "empty": empty}, fh)
        except Exception as exc:  # read-only filesystem etc. — cache is only an optimization
            logger.warning("Could not save vector cache: %s", exc)

    # ── (re)loading ──

    def _load(self, force: bool = False) -> None:
        with self._lock:
            pdf_files = self._scan_pdfs()
            if not pdf_files:
                logger.warning("No PDFs found in %s", self.pdf_directory)
                self.pdf_files, self.indexed_files, self.empty_files = [], [], []
                self.vector_store = None
                return

            manifest = self._build_manifest(pdf_files)
            if not force:
                cached = self._try_load_cache(manifest)
                if cached:
                    store, indexed, empty = cached
                    self.pdf_files, self.indexed_files, self.empty_files = pdf_files, indexed, empty
                    self.vector_store = store
                    return

            documents, indexed, empty = [], [], []
            for filename in pdf_files:
                text, n_pages, n_text = self._read_pdf(os.path.join(self.pdf_directory, filename))
                if text.strip():
                    per_page = len(text) // max(n_pages, 1)
                    if n_pages and (n_text / n_pages < 0.8 or per_page < 500):
                        logger.warning("%s: little text for its length — scanned or partial?", filename)
                    indexed.append(filename)
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                else:
                    logger.warning("No text extracted from %s — it will NOT be searchable.", filename)
                    empty.append(filename)

            if not documents:
                self.pdf_files, self.indexed_files, self.empty_files = pdf_files, [], empty
                self.vector_store = None
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Vector store ready: %d chunks from %d PDFs.", len(chunks), len(indexed))

            self.pdf_files, self.indexed_files, self.empty_files = pdf_files, indexed, empty
            self.vector_store = store
            self._save_cache(store, manifest, indexed, empty)

    def reload(self, force: bool = False) -> str:
        self._load(force=force)
        msg = (f"{'Reconstruido' if force else 'Recargado'}: {len(self.pdf_files)} PDF(s) en disco, "
               f"{len(self.indexed_files)} consultables.")
        if self.empty_files:
            msg += "\n\n⚠️ Sin texto extraíble (¿escaneados?):\n" + "\n".join(f"- {f}" for f in self.empty_files)
        return msg

    # ── querying ──

    @staticmethod
    def _to_cosine(sq_l2: float) -> float:
        """Squared L2 between unit vectors -> cosine similarity."""
        return max(-1.0, min(1.0, 1.0 - sq_l2 / 2.0))

    def query(self, query: str) -> Tuple[str, List[str]]:
        store = self.vector_store
        if store is None:
            return "", []
        try:
            k = min(int(store.index.ntotal), MAX_CANDIDATES)
        except Exception:
            k = MAX_CANDIDATES
        raw = store.similarity_search_with_score(query, k=max(k, 1))
        scored = [(doc, self._to_cosine(float(d))) for doc, d in raw]
        scored.sort(key=lambda x: x[1], reverse=True)

        kept = select_chunks(scored, SIMILARITY_THRESHOLD, PER_SOURCE_MIN, TOP_K,
                             MAX_CONTEXT_TOKENS, _CHARS_PER_TOKEN)
        hits, sources = [], []
        for doc, _s in kept:
            source = doc.metadata.get("source", "?")
            hits.append(f"[{source}]\n{doc.page_content.strip()}")
            if source not in sources:
                sources.append(source)
        logger.info("query=%r -> %d chunks from %d source(s)", query[:100], len(hits), len(sources))
        return ("\n\n---\n\n".join(hits) if hits else ""), sources

    def status_lines(self) -> List[str]:
        return [
            f"{'⚠️' if f in self.empty_files else '📄'} {f}"
            + ("  (sin texto, no se puede consultar)" if f in self.empty_files else "")
            for f in self.pdf_files
        ]


# ── conversation (one per browser session) ─────────────────────────────────────

class Conversation:
    """Last 6 turns (3 exchanges), token-capped, always starting with a 'user' turn."""

    MAX_HISTORY_TOKENS = 3000

    def __init__(self, max_history: int = 6):
        self.max_history = max_history
        self.turns: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        while len(self.turns) > self.max_history:
            self.turns.pop(0)
        while self.turns and self.turns[0]["role"] != "user":
            self.turns.pop(0)

    def get(self) -> List[Dict[str, str]]:
        total, trimmed = 0, []
        for msg in reversed(self.turns):
            total += len(msg["content"]) / _CHARS_PER_TOKEN
            if total > self.MAX_HISTORY_TOKENS:
                break
            trimmed.insert(0, msg)
        while trimmed and trimmed[0]["role"] != "user":
            trimmed.pop(0)
        return trimmed

    def last_user_message(self) -> Optional[str]:
        for msg in reversed(self.turns):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def clear(self) -> None:
        self.turns = []


# ── prompts + answering ────────────────────────────────────────────────────────

_BASE_SYSTEM = (
    "Eres un asistente académico especializado en Psicología del Desarrollo. "
    "Respeta el lenguaje, la terminología y los conceptos tal como los usan los autores. "
    "Cuando uses información de un texto específico, cita el nombre del archivo fuente. "
    "Si la pregunta involucra varios temas, integra información de múltiples fuentes cuando estén disponibles. "
    "Sé conciso y preciso: responde lo esencial sin extenderte innecesariamente. "
    "Al final de tu respuesta, en una línea nueva, agrega exactamente:\n"
    "[FILENAME: nombre_descriptivo.pdf]\n"
    "El nombre debe describir el contenido, usar guiones bajos, terminar en .pdf, "
    "y tener entre 5 y 50 caracteres."
)


def build_system_prompt(context_text: str) -> str:
    if context_text:
        return (
            _BASE_SYSTEM
            + "\n\nResponde utilizando exclusivamente los fragmentos de los textos "
            "académicos provistos a continuación. Cada fragmento está precedido por el "
            "nombre de su archivo entre corchetes: usa ese nombre para identificar de "
            "qué texto proviene (el título del texto o del capítulo puede no aparecer "
            "dentro del fragmento; por ejemplo, 'Seminario 3. Metáfora y Metonimia' "
            "corresponde al capítulo 'Metáfora y metonimia'). Los fragmentos provienen "
            "de una búsqueda semántica e incluyen los mejores de cada documento, por lo "
            "que algunos pueden no ser pertinentes: revísalos todos con atención antes "
            "de concluir. Solo si, tras leerlos, realmente no contienen la información "
            "pedida, indícalo con claridad en lugar de inventar.\n\n"
            "FRAGMENTOS DE LOS TEXTOS:\n\n" + context_text
        )
    return (
        _BASE_SYSTEM
        + "\n\nNo hay documentos con texto consultable cargados. "
        "Responde con tu conocimiento general e indica que no se pudo consultar ningún documento."
    )


def retrieval_query(conv: Conversation, user_message: str) -> str:
    """Short follow-ups borrow the previous question for retrieval only."""
    if len(user_message.split()) < _FOLLOWUP_MAX_WORDS:
        prev = conv.last_user_message()
        if prev:
            return f"{prev}\n{user_message}"
    return user_message


def answer(client, model: str, store: DocumentStore, conv: Conversation, user_message: str) -> dict:
    """Retrieve -> prompt -> Claude -> clean up. Returns a dict for the UI."""
    context_text, sources = store.query(retrieval_query(conv, user_message))

    kwargs = dict(
        model=model,
        max_tokens=7500,
        system=build_system_prompt(context_text),
        messages=conv.get() + [{"role": "user", "content": user_message}],
    )
    if TEMPERATURE.strip():
        kwargs["temperature"] = float(TEMPERATURE)
    try:
        response = client.messages.create(**kwargs)
    except TypeError as exc:
        if "temperature" not in kwargs or "temperature" not in str(exc):
            raise
        logger.warning("SDK rejected 'temperature'; retrying without it.")
        kwargs.pop("temperature")
        response = client.messages.create(**kwargs)
    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    logger.info("model=%s tokens in=%d out=%d", model, response.usage.input_tokens, response.usage.output_tokens)

    text, filename = split_filename(raw, user_message)
    if context_text:  # don't remember "no documents" exchanges
        conv.add("user", user_message)
        conv.add("assistant", text)

    pdf_bytes = build_response_pdf(user_message, text, sources).getvalue()
    return {"content": text, "sources": sources, "filename": filename, "pdf": pdf_bytes}
