from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    ROOT
    / "policy_query"
    / "policy.index"
)

CHUNKS_PATH = (
    ROOT
    / "policy_query"
    / "chunks.json"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Evaluation queries
# ---------------------------------------------------------
EVALUATION_QUERIES = [
    {
        "query": "How many days do I have to return an apparel or footwear product?",
        "relevant_documents": [
            "policy_apparel_footwear_returns"
        ],
    },
    {
        "query": "What is the return policy for electronics?",
        "relevant_documents": [
            "policy_electronics_returns"
        ],
    },
    {
        "query": "Can I return a home product?",
        "relevant_documents": [
            "policy_home_returns"
        ],
    },
    {
        "query": "How are COD refunds processed?",
        "relevant_documents": [
            "policy_cod_refunds"
        ],
    },
    {
        "query": "What is the expected delivery time?",
        "relevant_documents": [
            "policy_delivery_sla"
        ],
    },
    {
        "query": "What should I do if my product is damaged?",
        "relevant_documents": [
            "policy_damaged_item"
        ],
    },
    {
    "query": "Does the product need to be unused and undamaged with original packaging for a return?",
    "relevant_documents": [
        "policy_return_condition"
    ],
},
    {
        "query": "When is the refund processed after pickup?",
        "relevant_documents": [
            "policy_refund_after_pickup"
        ],
    },
]


def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def retrieve(
    query,
    model,
    index,
    chunks,
    k=3,
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "score": float(score),
                "document_id": chunk[
                    "document_id"
                ],
                "title": chunk.get(
                    "title",
                    "",
                ),
                "text": chunk["text"],
            }
        )

    return results


def precision_at_3(
    retrieved_ids,
    relevant_ids,
):

    retrieved_top3 = retrieved_ids[:3]

    hits = sum(
        1
        for doc_id in retrieved_top3
        if doc_id in relevant_ids
    )

    return hits / 3


def recall_at_3(
    retrieved_ids,
    relevant_ids,
):

    retrieved_top3 = retrieved_ids[:3]

    hits = sum(
        1
        for doc_id in retrieved_top3
        if doc_id in relevant_ids
    )

    if not relevant_ids:
        return 0.0

    return hits / len(
        relevant_ids
    )


def main():

    print("=" * 60)
    print("POLICY RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    chunks = load_chunks()

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    precision_scores = []
    recall_scores = []

    print(
        f"Evaluating {len(EVALUATION_QUERIES)} queries..."
    )

    for item in EVALUATION_QUERIES:

        query = item["query"]

        relevant_ids = set(
            item["relevant_documents"]
        )

        results = retrieve(
            query,
            model,
            index,
            chunks,
            k=3,
        )

        retrieved_ids = []

        for result in results:
            document_id = result["document_id"]

            if document_id not in retrieved_ids:
                retrieved_ids.append(document_id)

        precision = precision_at_3(
            retrieved_ids,
            relevant_ids,
        )

        recall = recall_at_3(
            retrieved_ids,
            relevant_ids,
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        print()
        print("Query:", query)
        print(
            "Expected:",
            sorted(relevant_ids),
        )
        print(
            "Retrieved:",
            retrieved_ids,
        )
        print(
            f"Precision@3: {precision:.3f}"
        )
        print(
            f"Recall@3: {recall:.3f}"
        )

    mean_precision = float(
        np.mean(
            precision_scores
        )
    )

    mean_recall = float(
        np.mean(
            recall_scores
        )
    )

    print()
    print("=" * 60)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 60)

    print(
        f"Mean Precision@3: {mean_precision:.3f}"
    )

    print(
        f"Mean Recall@3:    {mean_recall:.3f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()