# Chat API (Stateless `/chat` Endpoint)

This document describes the **stateless chat API** exposed by the FastAPI app in `backend/main.py`, how it connects to the `ChatManager` / `run_pipeline` orchestrator in `backend/chat/chat_manager.py`, and how to call it from external clients (curl, Python, etc.).

It is focused on the `/chat` endpoint used by `frontend/chat.html`.

---

## Table of Contents

1. [What this README covers](#1-what-this-readme-covers)  
2. [High-level data flow](#2-high-level-data-flow)  
3. [Request schema](#3-request-schema)  
   3.1. [Top-level request body](#31-top-level-request-body)  
   3.2. [`params` contract](#32-params-contract)  
4. [Backend defaults (`Settings`)](#4-backend-defaults-settings)  
5. [Response shape](#5-response-shape)  
6. [Example calls](#6-example-calls)  
   6.1. [Curl example (minimal)](#61-curl-example-minimal)  
   6.2. [Curl example with processing steps hidden](#62-curl-example-with-processing-steps-hidden)  
   6.3. [Python example using `requests`](#63-python-example-using-requests)  
7. [Notes for integrators](#7-notes-for-integrators)  

---

## 1. What this README covers

- **Scope**
  - The `POST /chat` **stateless** endpoint.
  - How the request maps to `ChatRequest`, `handle_chat`, and `run_pipeline`.
  - The JSON request/response shape, including the `params` contract.
  - How to control **processing stage streaming** via `show_processing_steps`.

- **Out of scope**
  - Ingest endpoints (`/index`, `/mediawiki/url`, `/pdf`, etc.).
  - Stateful endpoints such as `/chat/{session_id}` and session management.
  - SSE streaming internals (`backend/stream_stages.py`, `backend/stream_emit.py`).

---

## 2. High-level data flow

1. **Client (browser / external caller)**  
   Sends `POST /chat` with a JSON body matching `backend/core/schemas.ChatRequest`.

2. **FastAPI (`backend/main.py`)**  
   - Route:
     ```python
     @app.post("/chat", tags=["3. Search & Chat"], summary="5. Chat (stateless)")
     async def chat_with_content(chat_request: ChatRequest):
         ...
     ```
   - Delegates to `ChatManager.handle_chat` (or the module-level `handle_chat`):
     ```python
     result = await asyncio.to_thread(handler, chat_request.model_dump())
     return result
     ```

3. **Chat manager (`backend/chat/chat_manager.py`)**
   - `handle_chat(payload: Dict[str, Any]) -> Dict[str, Any]`:
     - Extracts `message`, `history`, and `params` from the payload.
     - Prepares `deps` and `req`.
     - Calls `run_pipeline(deps=deps, req=req)`.
   - `run_pipeline(...)` contains the full RAG pipeline:
     - query rewrite → retrieve → maybe rerank → summarize history → build prompt → inference → optional tools → final answer + metrics.

4. **Response**
   - `handle_chat` returns a JSON dict shaped like a `ChatResponse`-plus-metrics:
     ```json
     {
       "answer": "...",
       "response": "...",
       "metrics": { ... },
       "turn_metrics": { ... },
       "conversation_totals": { ... },
       "tools_used": [ ... ],
       "rewrite_display": { ... }
     }
     ```

---

## 3. Request schema

### 3.1 Top-level request body

The `/chat` route uses `backend.core.schemas.ChatRequest`:

```python
class ChatRequest(BaseModel):
    message: str
    context: List[Dict] = []
    use_web_search: bool = False
    # Pass-through of UI parameters and chat bubbles history (stateless UI)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
```

- **`message`**  
  Users current query (required).

- **`context`**  
  Reserved for future use; not required for stateless path.

- **`use_web_search`**  
  Currently not used by the stateless HTML chat path (`chat.html` sets this to `false`).

- **`params`**  
  Arbitrary dict of pipeline parameters, passed through to `run_pipeline`.

- **`history`**  
  Optional list of prior bubbles:
  ```json
  [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
  ```

---

### 3.2 `params` contract

`params` is a **flat dictionary**. Common keys:

#### Retrieval

- `top_k: int | null`  
- `score_threshold: float | null`

#### Summarizer / history window

- `chat_history_window_turns: int | null`  
- `raw_tail_turns: int | null`  
- `summarizer_max_input_tokens: int | null`  
- `summarizer_max_output_tokens: int | null`

#### Inference

- `temperature: float | null`  
- `top_p: float | null`  
- `max_output_tokens: int | null`

#### Query rewrite

- `enable_query_rewrite: bool | null`  
- `rewrite_confidence_threshold: float | null`  
- `rewrite_tail_turns: int | null`

#### Tools

- `use_tools: bool`

#### Provider/model overrides (optional)

- `inference_provider`, `inference_model`  
- `rewrite_provider`, `rewrite_model`  
- `summary_provider`, `summary_model`  
- `rerank_provider`, `rerank_model`

#### UX / observability

- `query_id: str`  
- `conversation_id: str`

#### Processing-stage visibility

- `show_processing_steps: bool`  

Controls intermediate SSE stage events (query rewrite, retrieval, rerank, summary, web context, prompt build, generating response, tool calls, tool synthesis). Final `"Final Answer"` and `"Done"` stages are always emitted.

---

## 4. Backend defaults (`Settings`)

`backend/core/config.py`:

```python
class Settings(BaseSettings):
    ...
    # 11) Debug / logging controls
    debug_verbose: bool = False
    debug_log_keys: bool = False
    debug_log_truncate_chars: int = 200  # max chars to print when debug_verbose is True
    show_processing_steps: bool = True  # controls whether intermediate SSE processing stages are emitted
```

Resolution in `run_pipeline`:

1. If `params["show_processing_steps"]` exists → use that.  
2. Else → fall back to `settings.show_processing_steps`.

---

## 5. Response shape

Typical response:

```json
{
  "answer": "Final answer text",
  "response": "Final answer text",
  "metrics": {
    "vectors_retrieved": 8
  },
  "turn_metrics": { },
  "conversation_totals": { },
  "tools_used": ["get_weather"],
  "rewrite_display": {
    "enabled": true,
    "triggered": true,
    "accepted": true,
    "original": "Where is it?",
    "rewritten": "Where is Mount Whitney located?",
    "confidence": 0.82,
    "threshold": 0.67,
    "ambiguous": false,
    "reason": "",
    "changed": true
  }
}
```

---

## 6. Example calls

### 6.1. Curl example (minimal)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain how this RAG chat pipeline works.",
    "use_web_search": false,
    "history": [],
    "params": {
      "top_k": 8,
      "score_threshold": 0.35,
      "temperature": 0.4,
      "max_output_tokens": 300,
      "enable_query_rewrite": true,
      "rewrite_confidence_threshold": 0.67,
      "rewrite_tail_turns": 1,
      "use_tools": false,
      "show_processing_steps": true,
      "query_id": "abcd1234",
      "conversation_id": "demo-convo-1"
    }
  }'
```

### 6.2. Curl example with processing steps hidden

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain how this RAG chat pipeline works.",
    "use_web_search": false,
    "history": [],
    "params": {
      "top_k": 8,
      "score_threshold": 0.35,
      "temperature": 0.4,
      "max_output_tokens": 300,
      "enable_query_rewrite": true,
      "rewrite_confidence_threshold": 0.67,
      "rewrite_tail_turns": 1,
      "use_tools": false,
      "show_processing_steps": false,
      "query_id": "abcd1234",
      "conversation_id": "demo-convo-1"
    }
  }'
```

### 6.3. Python example using `requests`

```python
import uuid
import requests

BASE_URL = "http://localhost:8000"

def call_chat(message: str, show_steps: bool = True):
    query_id = uuid.uuid4().hex[:8]
    conversation_id = "demo-conversation-1"

    payload = {
        "message": message,
        "use_web_search": False,
        "history": [],
        "params": {
            "top_k": 8,
            "score_threshold": 0.35,
            "summarizer_max_input_tokens": 400,
            "summarizer_max_output_tokens": 200,
            "chat_history_window_turns": 2,
            "raw_tail_turns": 2,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 300,
            "enable_query_rewrite": True,
            "rewrite_confidence_threshold": 0.67,
            "rewrite_tail_turns": 1,
            "use_tools": False,
            "show_processing_steps": show_steps,
            "query_id": query_id,
            "conversation_id": conversation_id,
        },
    }

    resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print("Answer:", data.get("answer") or data.get("response"))
    print("Metrics:", data.get("metrics"))
    print("Turn metrics:", data.get("turn_metrics"))
    print("Conversation totals:", data.get("conversation_totals"))
    print("Tools used:", data.get("tools_used"))
    print("Rewrite display:", data.get("rewrite_display"))

if __name__ == "__main__":
    call_chat("Give me a short overview of how this RAG chat pipeline works.", show_steps=True)
```

---

## 7. Notes for integrators

- `/chat` is ideal for:
  - Browser-based UIs similar to `frontend/chat.html`.
  - External clients that manage their own `conversation_id` and `history`.

- Use `params.show_processing_steps` for **per-turn** control of intermediate stage visibility, and `settings.show_processing_steps` (or `SHOW_PROCESSING_STEPS` env) for global defaults.

- The RAG logic and final answer are unchanged by `show_processing_steps`; it only affects whats emitted on the SSE "Processing steps" stream.
