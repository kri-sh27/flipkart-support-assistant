from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

KB_PATH = ROOT / "policy_query" / "policy_kb.json"
INDEX_PATH = ROOT / "policy_query" / "policy.index"
CHUNKS_PATH = ROOT / "policy_query" / "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_policy_documents():
    with open(
        KB_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def create_sentence_chunks(documents):
    chunks = []

    for document in documents:
        document_id = document["id"]
        title = document.get("title", "")
        text = document["text"]

        # Sentence-level chunking
        sentences = [
            sentence.strip()
            for sentence in text.replace("!", ".")
            .replace("?", ".")
            .split(".")
            if sentence.strip()
        ]

        for sentence in sentences:
            chunks.append(
    {
        "document_id": document_id,
        "title": title,
        "text": sentence,
        "embedding_text": f"{title}. {sentence}",
    }
)

    return chunks


def build_faiss_index(chunks):
    texts = [
    f"{chunk['title']}. {chunk['text']}"
    for chunk in chunks
]

    print(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def main():

    print("=" * 60)
    print("FLIPKART POLICY RAG INDEX BUILDER")
    print("=" * 60)

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Policy KB not found: {KB_PATH}"
        )

    documents = load_policy_documents()

    print(
        f"Loaded {len(documents)} policy documents."
    )

    if len(documents) < 12:
        raise ValueError(
            f"At least 12 policy documents are required. "
            f"Found: {len(documents)}"
        )

    chunks = create_sentence_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} sentence-level chunks."
    )

    index = build_faiss_index(
        chunks
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Index created successfully.")
    print(f"Index:  {INDEX_PATH}")
    print(f"Chunks: {CHUNKS_PATH}")
    print(
        f"FAISS vectors: {index.ntotal}"
    )


if __name__ == "__main__":
    main()