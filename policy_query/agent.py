from pathlib import Path
import json
import re
from typing import TypedDict, Optional, List, Dict, Any

import faiss
import numpy as np
from joblib import load
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

from product_image_query.predict_product import predict as classify_product_image


ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Part 1 - Return Risk Model
# ============================================================

MODEL = load(
    ROOT / "models" / "return_risk_model.pkl"
)

T_RF = json.loads(
    (
        ROOT
        / "models"
        / "return_risk_threshold.json"
    ).read_text(
        encoding="utf-8"
    )
)["t_rf"]


# ============================================================
# Part 3 - Policy RAG
# ============================================================

INDEX = faiss.read_index(
    str(
        ROOT
        / "policy_query"
        / "policy.index"
    )
)

CHUNKS = json.loads(
    (
        ROOT
        / "policy_query"
        / "chunks.json"
    ).read_text(
        encoding="utf-8"
    )
)

EMBED = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LangGraph State
# ============================================================

class State(TypedDict, total=False):
    messages: List[Dict[str, str]]
    order_id: Optional[str]
    intent: str
    retrieved: List[Dict[str, Any]]
    tool_output: Dict[str, Any]
    answer: Dict[str, Any]
    blocked: bool


# ============================================================
# Prompt Injection Guard
# ============================================================

INJECTION = re.compile(
    r"ignore\s+(previous|all)\s+(instructions|rules)"
    r"|pretend\s+you\s+are"
    r"|disregard\s+(previous|all)"
    r"|reveal\s+(system|hidden)\s+(prompt|instructions)"
    r"|bypass\s+(the\s+)?(rules|guardrails)",
    re.I,
)



def intent_node(s):

    messages = s.get("messages", [])

    if not messages:
        return {
            "intent": "policy"
        }

    latest_text = messages[-1]["content"]

    # --------------------------------------------------------
    # Prompt injection check
    # --------------------------------------------------------

    if INJECTION.search(latest_text):
        return {
            "blocked": True,
            "intent": "blocked",
        }

    # --------------------------------------------------------
    # Recover order ID from conversation state
    # --------------------------------------------------------

    order_id = s.get("order_id")

    for message in messages:

        content = message.get(
            "content",
            ""
        )

        if re.search(
        r"\border\s+(?:id|number)\b",
        content,
        re.I,
    ):
            continue

        match = re.search(
            r"\border\s*(?:#\s*)?(?!id\b|number\b)([A-Za-z0-9-]+)\b",
            content,
            re.I,
        )

        if match:
            order_id = match.group(1)

    lower_text = latest_text.lower()

    # --------------------------------------------------------
    # Explicit order-ID / state query
    # --------------------------------------------------------

    if (
        "order id" in lower_text
        or "order number" in lower_text
    ):

        return {
            "intent": "order_lookup",
            "order_id": order_id,
        }

    # --------------------------------------------------------
    # Return-risk routing
    # --------------------------------------------------------

    if any(
        keyword in lower_text
        for keyword in [
            "return risk",
            "risk of return",
            "likely to return",
            "return probability",
        ]
    ):

        return {
            "intent": "return_risk",
            "order_id": order_id,
        }

    # --------------------------------------------------------
    # Product-image routing
    # --------------------------------------------------------

    elif any(
        keyword in lower_text
        for keyword in [
            "image",
            "product category",
            "what category",
            "classify",
        ]
    ):

        return {
            "intent": "product_category",
            "order_id": order_id,
        }

    # --------------------------------------------------------
    # Default = policy
    # --------------------------------------------------------

    return {
        "intent": "policy",
        "order_id": order_id,
    }
# ============================================================
# Policy Retrieval Node
# ============================================================

def retrieve_node(s):

    if s.get("intent") != "policy":
        return {
            "retrieved": []
        }

    query = s["messages"][-1]["content"]

    vector = EMBED.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = INDEX.search(
        vector,
        3,
    )

    retrieved = []

    for score, chunk_id in zip(
        scores[0],
        ids[0],
    ):

        if chunk_id < 0:
            continue

        chunk = CHUNKS[int(chunk_id)]

        retrieved.append(
            {
                "score": float(score),
                **chunk,
            }
        )

    return {
        "retrieved": retrieved
    }


# ============================================================
# Part 1 - Return Risk Tool
# ============================================================

def check_return_risk(order_features):

    import pandas as pd

    features = pd.DataFrame(
        [order_features]
    )

    probability = float(
        MODEL.predict_proba(
            features
        )[0, 1]
    )

    if probability < T_RF:
        bucket = "Low"

    elif probability >= T_RF + 0.15:
        bucket = "High"

    else:
        bucket = "Medium"

    return {
        "probability": probability,
        "risk_bucket": bucket,
        "t_rf": T_RF,
        "cut_points": {
            "low_lt": T_RF,
            "high_gte": T_RF + 0.15,
        },
    }


# ============================================================
# Tool Node
# ============================================================

def tool_node(s):

    # --------------------------------------------------------
    # Return Risk
    # --------------------------------------------------------

    if s.get("intent") == "return_risk":

        text = s["messages"][-1]["content"]

        match = re.search(
            r"FEATURES:\s*(\{.*\})",
            text,
            re.S,
        )

        if not match:
            return {
                "tool_output": {
                    "error": (
                        "Provide order features as JSON "
                        "after FEATURES:"
                    )
                }
            }

        try:

            features = json.loads(
                match.group(1)
            )

            return {
                "tool_output": check_return_risk(
                    features
                )
            }

        except Exception as exc:

            return {
                "tool_output": {
                    "error": (
                        f"Invalid order features: {exc}"
                    )
                }
            }

    # --------------------------------------------------------
    # Product Image Classification
    # --------------------------------------------------------

    if s.get("intent") == "product_category":

        text = s["messages"][-1]["content"]

        match = re.search(
            r"IMAGE:\s*(\S+)",
            text,
        )

        if not match:

            return {
                "tool_output": {
                    "error": (
                        "Provide IMAGE: path"
                    )
                }
            }

        image_path = match.group(1)

        path = Path(image_path)

        if not path.is_absolute():

            path = ROOT / image_path

        try:

            result = classify_product_image(
                str(path)
            )

            return {
                "tool_output": result
            }

        except Exception as exc:

            return {
                "tool_output": {
                    "error": (
                        f"Image classification failed: {exc}"
                    )
                }
            }

    return {
        "tool_output": {}
    }




def check_groundedness(retrieved, threshold=0.50):
    """
    Verify that the policy response has sufficient
    supporting evidence from the policy knowledge base.
    """

    grounded = [
        item
        for item in retrieved
        if float(item.get("score", 0.0)) >= threshold
    ]

    if not grounded:
        return {
            "grounded": False,
            "confidence": 0.0,
            "evidence": [],
        }

    return {
        "grounded": True,
        "confidence": float(grounded[0]["score"]),
        "evidence": grounded,
    }

def response_node(s):

    # --------------------------------------------------------
    # Prompt injection blocked
    # --------------------------------------------------------

    if s.get("blocked"):

        answer = {
            "answer": (
                "I can't follow instructions that attempt "
                "to override the support assistant's rules. "
                "Please ask a normal support question."
            ),
            "source": "guardrail",
            "confidence": 1.0,
        }

    elif s.get("intent") == "order_lookup":

        order_id = s.get("order_id")

        if order_id:

            answer = {
                "answer": (
                    f"The order ID mentioned in this "
                    f"conversation is {order_id}."
                ),
                "source": "conversation_state",
                "confidence": 1.0,
            }

        else:

            answer = {
                "answer": (
                    "I don't have an order ID available "
                    "in the current conversation."
                ),
                "source": "conversation_state",
                "confidence": 0.0,
            }

    # --------------------------------------------------------
    # Policy RAG response
    # --------------------------------------------------------

    elif s.get("intent") == "policy":

        # retrieved = s.get(
        #     "retrieved",
        #     []
        # )
        grounding = check_groundedness(
                s.get("retrieved", [])
            )
        if not grounding["grounded"]:

            answer = {
                "answer": (
                    "I don't have enough grounded information "
                    "in the policy knowledge base to answer "
                    "that reliably."
                ),
                "source": "groundedness_guardrail",
                "confidence": 0.0,
            }

        else:

            best = grounding["evidence"][0]

            answer = {
                "answer": best["text"],
                "source": best["document_id"],
                "confidence": grounding["confidence"],
            }
 

    # --------------------------------------------------------
    # Return Risk response
    # --------------------------------------------------------

    elif s.get("intent") == "return_risk":

        output = s.get(
            "tool_output",
            {}
        )

        if "error" in output:

            answer = {
                "answer": output["error"],
                "source": "return_risk_tool",
                "confidence": 0.0,
            }

        else:

            probability = float(
                output.get(
                    "probability",
                    0.0
                )
            )

            answer = {
                "answer": (
                    "The predicted return probability "
                    f"is {probability:.3f}, which is a "
                    f"{output.get('risk_bucket', 'Unknown')} "
                    "risk. "
                    f"The bucket is anchored to "
                    f"t*_rf={T_RF:.2f}."
                ),
                "source": "return_risk_tool",
                "confidence": probability,
            }

    # --------------------------------------------------------
    # Product classification response
    # --------------------------------------------------------

    else:

        output = s.get(
            "tool_output",
            {}
        )

        if "error" in output:

            answer = {
                "answer": output["error"],
                "source": "image_classifier_tool",
                "confidence": 0.0,
            }

        else:

            confidence = float(
                output.get(
                    "confidence",
                    0.0
                )
            )

            answer = {
                "answer": (
                    "The predicted product category "
                    f"is {output.get('category', 'Unknown')} "
                    f"with confidence {confidence:.3f}."
                ),
                "source": "image_classifier_tool",
                "confidence": confidence,
            }

    return {
        "answer": answer
    }

# ============================================================
# Conditional Routing
# ============================================================

def route(s):

    return s.get(
        "intent",
        "policy",
    )


# ============================================================
# LangGraph Workflow
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "intent",
    intent_node,
)

graph.add_node(
    "retrieve",
    retrieve_node,
)

graph.add_node(
    "tools",
    tool_node,
)

graph.add_node(
    "response",
    response_node,
)

graph.add_edge(
    START,
    "intent",
)

graph.add_conditional_edges(
    "intent",
    route,
    {
        "policy": "retrieve",
        "return_risk": "tools",
        "product_category": "tools",
                "order_lookup": "response",

        "blocked": "response",
    },
)

graph.add_edge(
    "retrieve",
    "response",
)

graph.add_edge(
    "tools",
    "response",
)

graph.add_edge(
    "response",
    END,
)

GRAPH = graph.compile()


# ============================================================
# Public Agent Function
# ============================================================

def run(
    messages,
    order_id=None,
):

    return GRAPH.invoke(
        {
            "messages": messages,
            "order_id": order_id,
        }
    )


if __name__ == "__main__":

    print(
        "Flipkart Support Agent loaded successfully."
    )

    print(
        f"RF threshold: {T_RF:.4f}"
    )

    print(
        f"Policy chunks loaded: {len(CHUNKS)}"
    )

    print(
        f"FAISS vectors: {INDEX.ntotal}"
    )

    print(
        "LangGraph workflow compiled successfully."
    )