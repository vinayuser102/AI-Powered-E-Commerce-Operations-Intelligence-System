"""Ingest the SmartOps policy document into the shared ChromaDB collection."""

import logging
import os
import re
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base.txt"
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "chroma_storage"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "smartops_policies")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Create bounded, overlapping chunks while preferring paragraph boundaries."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def run_ingestion() -> int:
    """Upsert policy chunks with provenance metadata into the API's collection."""
    raw_text = KNOWLEDGE_BASE.read_text(encoding="utf-8")
    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("The knowledge base contains no ingestible content")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.upsert(
        documents=chunks,
        ids=[f"smartops_policy_{index}" for index in range(len(chunks))],
        metadatas=[{"source": KNOWLEDGE_BASE.name, "chunk": index} for index in range(len(chunks))],
    )
    logger.info("Indexed %d policy chunks in %s", len(chunks), COLLECTION_NAME)
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()
