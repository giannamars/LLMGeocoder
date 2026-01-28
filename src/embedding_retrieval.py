# src/embedding_retrieval.py
import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import numpy as np

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

VECTORSTORE_ROOT = Path("./vectorstores")
VECTORSTORE_ROOT.mkdir(parents=True, exist_ok=True)

_QUERY = (
    "Burkholderia pseudomallei melioidosis case patient hospital clinic admitted diagnosed treated location country "
    "environmental farm soil water sample collected site coordinates"
    "collected year date study city province district region"
)

DEFAULT_TOP_K = 3

# ----------------------------------------------------------------------
# Cached singletons (loaded once, reused)
# ----------------------------------------------------------------------
_cached_embedder: HuggingFaceEmbeddings | None = None
_cached_query_vec: np.ndarray | None = None


def _get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached embedder instance (loaded once)."""
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _cached_embedder


def _get_query_vec() -> np.ndarray:
    """Return the embedding of the static query; compute once."""
    global _cached_query_vec
    if _cached_query_vec is None:
        embedder = _get_embedder()
        _cached_query_vec = np.array(embedder.embed_query(_QUERY), dtype=np.float32)
    return _cached_query_vec


def _split(text: str) -> List[Document]:
    """Split raw text into overlapping chunks wrapped in Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
    return splitter.create_documents([text])


def build_index(pmid: str, full_text: str) -> bool:
    """
    Build (or rebuild) a persistent Chroma collection for a single PMID.
    Returns True on success, False on failure.
    """
    if not full_text:
        logging.warning(f"[{pmid}] Empty document – skipping indexing.")
        return False

    chunks = _split(full_text)
    if not chunks:
        logging.error(f"[{pmid}] No chunks produced – cannot build index.")
        return False

    dest_dir = VECTORSTORE_ROOT / pmid
    if dest_dir.is_dir():
        for p in dest_dir.rglob("*"):
            p.unlink()
        dest_dir.rmdir()

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=_get_embedder(),
            collection_name="metadata",
            persist_directory=str(dest_dir),
        )
    except Exception as exc:
        logging.error(f"[{pmid}] Failed to create Chroma collection: {exc}")
        return False

    logging.info(f"[{pmid}] Indexed {len(chunks)} chunks → {dest_dir}")
    return True


def _load_collection(pmid: str) -> Chroma | None:
    """Return a Chroma object for a PMID if it exists, else None."""
    coll_path = VECTORSTORE_ROOT / pmid
    if not coll_path.is_dir():
        logging.warning(f"Vector store missing for PMID {pmid}")
        return None

    return Chroma(
        persist_directory=str(coll_path),
        collection_name="metadata",
        embedding_function=_get_embedder(),
    )


def retrieve_top_chunks(pmid: str, top_k: int = DEFAULT_TOP_K) -> str | None:
    """
    Load the vector store for `pmid` and return concatenated top_k chunks.
    Returns None if collection doesn't exist or search yields no results.
    """
    vect = _load_collection(pmid)
    if vect is None:
        return None

    query_vec = _get_query_vec().tolist()
    results: List[Document] = vect.similarity_search_by_vector(query_vec, k=top_k)

    if not results:
        logging.warning(f"Similarity search returned no results for PMID {pmid}")
        return None

    return "\n\n".join(doc.page_content.strip() for doc in results if doc.page_content)