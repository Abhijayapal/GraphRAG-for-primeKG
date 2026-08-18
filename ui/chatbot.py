"""
Drug Repurposing Knowledge Graph — Chainlit Chatbot UI
=======================================================

Architecture:
  User message
    → Groq Llama 3 detects intent (repurpose / explain / query)
    → Calls our FastAPI backend (localhost:8000)
    → Injects raw JSON (scores + graph paths) into strict system prompt
    → Streams answer back: 1-line summary + ranked list with scores + paths

Run with:
    chainlit run ui/chatbot.py --port 8001

Requirements:
    pip install chainlit groq httpx
    GROQ_API_KEY in .env
"""

import os
import json
import httpx
import chainlit as cl
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Clients ────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Model ─────────────────────────────────────────────────────────────────────
GROQ_MODEL = "qwen/qwen3.6-27b"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a drug discovery assistant powered by a biomedical knowledge graph (PrimeKG, 120,000+ nodes, 8M+ edges).

STRICT RULES:
1. You MUST answer ONLY using the JSON data provided below from the knowledge graph API.
2. Do NOT use your own training knowledge about drugs or diseases.
3. If the answer is not in the provided data, say: "I cannot find this in the knowledge graph."
4. Never invent drug names, scores, or paths.

RESPONSE FORMAT (always follow exactly):
[One concise sentence explaining the overall finding.]

📊 Graph Evidence:
[For each result, one line:]
#N. [Drug Name] | Score: [score] | Path: [biological path]

[One final sentence about what this means for the disease.]
"""

# ── Intent detection prompt ───────────────────────────────────────────────────
INTENT_PROMPT = """Classify the user's question into exactly one of these intents:
- REPURPOSE: User wants novel/new/off-label drug candidates for a disease (e.g. "What drugs can be repurposed for COVID?")
- QUERY: User wants known treatments/connections for a disease or drug (e.g. "What drugs treat Alzheimer's?")
- EXPLAIN: User wants to understand how a specific drug relates to a specific disease (e.g. "How does Heparin relate to COVID?")
- UNKNOWN: None of the above

Return ONLY a JSON object with these keys:
{
  "intent": "REPURPOSE" | "QUERY" | "EXPLAIN",
  "disease": "extracted disease name or null",
  "drug": "extracted drug name or null"
}

No explanation. JSON only."""


def detect_intent(user_message: str) -> dict:
    """Use Groq to classify intent and extract entities from user message."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=2000,
    )
    content = response.choices[0].message.content or ""
    import re
    raw = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "QUERY", "disease": None, "drug": None}


def call_api(intent: dict, user_message: str) -> dict:
    """Call the correct FastAPI endpoint based on detected intent."""
    action = intent.get("intent", "QUERY")
    disease = intent.get("disease") or ""
    drug = intent.get("drug") or ""

    try:
        if action == "REPURPOSE" and disease:
            resp = httpx.post(
                f"{API_BASE}/repurpose",
                json={"disease_name": disease, "top_k": 10},
                timeout=30,
            )
            return {"action": "REPURPOSE", "disease": disease, "data": resp.json()}

        elif action == "EXPLAIN" and drug and disease:
            resp = httpx.post(
                f"{API_BASE}/explain",
                json={"drug": drug, "disease": disease},
                timeout=30,
            )
            return {"action": "EXPLAIN", "drug": drug, "disease": disease, "data": resp.json()}

        else:
            # Default: QUERY (hybrid known-connection search)
            entity = disease or drug or user_message
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"name": entity, "entity_type": "disease" if disease else "drug", "top_k": 10},
                timeout=30,
            )
            return {"action": "QUERY", "entity": entity, "data": resp.json()}

    except httpx.ConnectError:
        return {"error": "Cannot connect to API. Make sure `uvicorn api.main:app` is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def format_api_data(api_result: dict) -> str:
    """Convert raw API JSON into a clean text block for the LLM prompt."""
    if "error" in api_result:
        return f"API ERROR: {api_result['error']}"

    data = api_result.get("data", {})
    if "detail" in data:
        # FastAPI error (e.g. 404 Entity not found)
        return f"Database Error: {data['detail']}"

    action = api_result.get("action")
    lines = []

    if action == "REPURPOSE":
        results = data.get("result", {}).get("predictions", [])
        lines.append(f"Novel drug candidates for: {api_result.get('disease')}")
        for i, r in enumerate(results[:10], 1):
            path = r.get("bio_path", r.get("path", "No direct path found"))
            if not path:
                path = "No direct path found"
            score = r.get("final_score", r.get("rotate_score", "N/A"))
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            lines.append(f"{i}. {r.get('drug', 'Unknown')} | Score: {score_str} | Path: {path}")

    elif action == "EXPLAIN":
        lines.append(f"Biological connection: {api_result.get('drug')} → {api_result.get('disease')}")
        direct = data.get("direct", [])
        indirect = data.get("indirect", [])
        if direct:
            lines.append(f"Direct edge found: {direct[0]}")
        for path in indirect[:5]:
            lines.append(f"Indirect path: {path}")
        if not direct and not indirect:
            lines.append("No direct or indirect path found in the graph.")

    elif action == "QUERY":
        results_dict = data.get("results", {})
        candidates = results_dict.get("candidates", []) if isinstance(results_dict, dict) else results_dict
        lines.append(f"Hybrid search results for: {api_result.get('entity')}")
        for i, r in enumerate(candidates[:10], 1):
            score = r.get("hybrid_score", r.get("score", "N/A"))
            path = r.get("path_str", r.get("path", "No explicit path"))
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            lines.append(f"{i}. {r.get('name', r.get('drug', 'Unknown'))} | Score: {score_str} | Path: {path}")

    return "\n".join(lines) if lines else "No results returned from the graph."


def stream_llm_answer(user_message: str, graph_data_text: str):
    """Stream a grounded Groq Llama 3 answer using only graph data."""
    user_content = f"""User question: {user_message}

Knowledge graph data:
{graph_data_text}

Answer strictly using the format specified. Use ONLY the above data."""

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=2000,
        stream=True,
    )
    in_think = False
    buffer = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        buffer += token
        
        while buffer:
            if not in_think:
                idx = buffer.find("<think>")
                if idx != -1:
                    if idx > 0:
                        yield buffer[:idx]
                    buffer = buffer[idx + 7:]
                    in_think = True
                else:
                    if any(buffer.endswith(p) for p in ["<", "<t", "<th", "<thi", "<thin", "<think"]):
                        break
                    yield buffer
                    buffer = ""
                    break
            else:
                idx = buffer.find("</think>")
                if idx != -1:
                    buffer = buffer[idx + 8:]
                    in_think = False
                else:
                    buffer = ""
                    break
    if not in_think and buffer and not any(buffer.startswith(p) for p in ["<", "<t", "<th", "<thi", "<thin", "<think"]):
        yield buffer


# ── Chainlit event handlers ───────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Greet the user and check API health."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        status = resp.json().get("status", "unknown")
        api_ok = status == "ok"
    except Exception:
        api_ok = False

    api_icon = "✅" if api_ok else "❌"
    await cl.Message(
        content=(
            f"## 💊 Drug Repurposing Knowledge Graph\n\n"
            f"**Backend API:** {api_icon} {'Connected' if api_ok else 'Not connected — run `uvicorn api.main:app`'}\n\n"
            f"Ask me anything about drug-disease relationships. Examples:\n"
            f"- *\"What are novel drug candidates for COVID-19?\"*\n"
            f"- *\"What drugs treat Alzheimer's disease?\"*\n"
            f"- *\"How does Heparin relate to COVID-19?\"*\n\n"
            f"Every answer includes the biological graph path and confidence scores."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Main handler: route → call API → stream grounded LLM answer."""
    user_text = message.content.strip()

    # Step 1: Show thinking indicator
    thinking_msg = cl.Message(content="🔍 Querying knowledge graph...")
    await thinking_msg.send()

    # Step 2: Detect intent
    intent = detect_intent(user_text)

    # Step 3: Call the right API endpoint
    api_result = call_api(intent, user_text)

    # Step 4: Format the raw graph data
    graph_data_text = format_api_data(api_result)

    # Step 5: Update status
    action_label = {
        "REPURPOSE": "🔬 Novel Link Prediction",
        "EXPLAIN": "🗺️ Cypher Path Extraction",
        "QUERY": "⚡ Hybrid Ranker (Cypher + RotatE)",
    }.get(api_result.get("action", "QUERY"), "⚡ Hybrid Ranker")

    await thinking_msg.update()

    # Step 6: Stream the grounded LLM answer
    response_msg = cl.Message(content=f"**Mode:** {action_label}\n\n")
    await response_msg.send()

    full_response = ""
    for token in stream_llm_answer(user_text, graph_data_text):
        full_response += token
        await response_msg.stream_token(token)

    await response_msg.update()
