# Chat API (Stateless `/chat` Endpoint)

> **About this document**
>
> This page provides **complete API documentation** for the *Chat-with-RAG* system, including request/response formats, parameters, and integration examples for the stateless chat endpoint.
>
> **Note:** If you landed here directly (for example from documentation hosting or search), start with the repository **[README](https://github.com/vrraj/chat-with-rag/blob/main/README.md)** to see how to run the system locally and try the interactive demo.

This document is focused on the `/chat` endpoint used by `frontend/chat.html`.

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
    use_web_search: Optional[bool] = None
    # Pass-through of UI parameters and chat bubbles history (stateless UI)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
```

- **`message`**  
  Users current query (required).

- **`context`**  
  Reserved for future use; not required for stateless path.

- **`use_web_search`**  
  Optional boolean for web search integration. Currently not used by the stateless HTML chat path (`chat.html` sets this to `null`).

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
- `namespace: str | null` - Domain/collection isolation

#### Summarizer / history window

- `raw_tail_turns: int | null`  
- `summarizer_max_input_tokens: int | null`  
- `summarizer_max_output_tokens: int | null`

#### Inference

- `temperature: float | null`  
- `top_p: float | null`  
- `max_output_tokens: int | null`
- `reasoning_effort: str | null` - Reasoning intensity for reasoning models ("minimal", "low", "medium", "high")
- `render_html: bool | null` - Enable server-side Markdown to HTML rendering

#### Query rewrite

- `enable_query_rewrite: bool | null`  
- `rewrite_confidence_threshold: float | null`  
- `rewrite_tail_turns: int | null`
- `rewrite_summary_turns: int | null`

#### Tools

- `use_tools: bool`

#### Provider/model overrides (optional)

- `model_keys: object` - Stage-specific model overrides:
  ```json
  {
    "inference": "gemini:gemini-2.5-flash",
    "rewrite": "openai:gpt-4o-mini",
    "rerank": "openai:gpt-4o-mini",
    "summary": "openai:gpt-4o-mini",
    "tools_synth": "openai:gpt-4o-mini"
  }
  ```

#### UX / observability

- `query_id: str`  
- `conversation_id: str`
- `user_id: str` - Optional user identifier for token accounting
- `show_sources: bool` - Source citation display control
- `prompt_domain: str` - Domain for prompt registry resolution

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
  "answer_html": "<p>Final answer with HTML formatting</p>",
  "reasoning": "Step-by-step reasoning process...",
  "metrics": {
    "vectors_retrieved": 8
  },
  "turn_metrics": {
    "tokens": {
      "embedding": 1500,
      "rewrite": 120,
      "rerank": 80,
      "inference": 250,
      "reasoning": 100,
      "total": 1950
    },
    "cost": {
      "embedding": 0.003,
      "rewrite": 0.002,
      "rerank": 0.001,
      "inference": 0.005,
      "total": 0.011
    }
  },
  "conversation_totals": {
    "tokens": {"total": 5000},
    "cost": {"total": 0.025},
    "messages": 3
  },
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
      "reasoning_effort": "low",
      "render_html": true,
      "enable_query_rewrite": true,
      "rewrite_confidence_threshold": 0.67,
      "rewrite_tail_turns": 1,
      "use_tools": false,
      "show_processing_steps": true,
      "show_sources": true,
      "namespace": "default",
      "prompt_domain": "default",
      "query_id": "abcd1234",
      "conversation_id": "demo-convo-1",
      "user_id": "user123",
      "model_keys": {
        "inference": "openai:gpt-4o-mini"
      }
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
      "rewrite_summary_turns": 3,
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
            "raw_tail_turns": 2,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 300,
            "reasoning_effort": "minimal",
            "render_html": False,
            "enable_query_rewrite": True,
            "rewrite_confidence_threshold": 0.67,
            "rewrite_tail_turns": 1,
            "rewrite_summary_turns": 3,
            "use_tools": False,
            "show_processing_steps": show_steps,
            "show_sources": True,
            "namespace": "default",
            "prompt_domain": "default",
            "query_id": query_id,
            "conversation_id": conversation_id,
            "user_id": "demo-user",
            "model_keys": {
                "inference": "openai:gpt-4o-mini",
                "rewrite": "openai:gpt-4o-mini",
                "rerank": "openai:gpt-4o-mini"
            }
        },
    }

    resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print("Answer:", data.get("answer") or data.get("response"))
    print("Answer HTML:", data.get("answer_html"))  # When HTML rendering is enabled
    print("Reasoning:", data.get("reasoning"))  # For reasoning models
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

- The RAG logic and final answer are unchanged by `show_processing_steps`; it only affects what's emitted on the SSE "Processing steps" stream.

---

# Session-Based Chat API (Stateful `/chat/{session_id}` Endpoint)

This document describes the **session-based chat API** that provides server-side conversation state management, ideal for backend integrations, mobile apps, and multi-device scenarios.

## Table of Contents

1. [Session API Overview](#1-session-api-overview)  
2. [Session Management](#2-session-management)  
   2.1. [Create Session](#21-create-session)  
   2.2. [Send Message to Session](#22-send-message-to-session)  
   2.3. [Get Session History](#23-get-session-history)  
3. [Model Override Examples](#3-model-override-examples)  
4. [Session vs Stateless Comparison](#4-session-vs-stateless-comparison)  

---

## 1. Session API Overview

The session-based API provides:
- **Server-side conversation state** - No need to send history in each request
- **Automatic context management** - Token-aware history truncation
- **Multi-device support** - Same session accessible from different clients
- **Identical pipeline quality** - Same RAG processing as stateless endpoint

### Key Differences from Stateless API

| Feature | Stateless (`/chat`) | Session-Based (`/chat/{session_id}`) |
|---------|-------------------|-------------------------------------|
| **History Management** | Client sends full history each request | Server maintains history automatically |
| **State** | No server state | Persistent session state |
| **Setup** | No setup required | Create session first |
| **Use Case** | Simple integrations, web UI | Backend systems, mobile apps |

---

## 2. Session Management

### 2.1. Create Session

**Endpoint:** `POST /chat/session`

**Request:** (empty body)

**Response:**
```json
{
  "session_id": "12d8cd79-0ee8-4dcd-97a5-5983effcbccd"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/chat/session
```

---

### 2.2. Send Message to Session

**Endpoint:** `POST /chat/{session_id}`

**Request Schema:** Same as stateless `/chat` endpoint, but `history` is optional (server manages it)

**Response Schema:** Same as stateless `/chat` endpoint

**Example - First Message:**
```bash
curl -X POST http://localhost:8000/chat/12d8cd79-0ee8-4dcd-97a5-5983effcbccd \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Mount Everest?",
    "history": [],
    "params": {
      "top_k": 5,
      "temperature": 0.7,
      "max_output_tokens": 500
    }
  }'
```

**Example - Follow-up Message (Context Preserved):**
```bash
curl -X POST http://localhost:8000/chat/12d8cd79-0ee8-4dcd-97a5-5983effcbccd \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How tall is it?",
    "history": [],
    "params": {
      "top_k": 5,
      "temperature": 0.7,
      "max_output_tokens": 500
    }
  }'
```

---

### 2.3. Get Session History

**Endpoint:** `GET /chat/{session_id}/history`

**Response:**
```json
{
  "session_id": "12d8cd79-0ee8-4dcd-97a5-5983effcbccd",
  "messages": [
    {"role": "user", "content": "What is Mount Everest?"},
    {"role": "assistant", "content": "Mount Everest is Earth's highest mountain..."},
    {"role": "user", "content": "How tall is it?"},
    {"role": "assistant", "content": "Mount Everest stands at 8,848 meters..."}
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T10:32:15Z"
}
```

**Example:**
```bash
curl http://localhost:8000/chat/12d8cd79-0ee8-4dcd-97a5-5983effcbccd/history
```

---

## 3. Model Override Examples

### 3.1. Override Inference Model

Use `model_keys` to override models per request:

```bash
curl -X POST http://localhost:8000/chat/fd91c243-1f0f-441a-8ce9-635377ba54a5 \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is the elevation difference with kilimanjaro?",
    "history": [],
    "params": {
      "top_k": 5,
      "temperature": 0.7,
      "max_output_tokens": 500,
      "model_keys": {
        "inference": "gemini:gemini-2.5-flash"
      }
    }
  }'
```

### 3.2. Stage-Specific Model Overrides

Override specific pipeline stages:

```bash
curl -X POST http://localhost:8000/chat/session-id \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "history": [],
    "params": {
      "top_k": 5,
      "temperature": 0.7,
      "max_output_tokens": 500,
      "model_keys": {
        "inference": "gemini:gemini-2.5-flash",
        "rewrite": "openai:gpt-4o-mini",
        "rerank": "openai:gpt-4o-mini",
        "summary": "openai:gpt-4o-mini",
        "tools_synth": "gemini:gemini-2.5-flash"
      }
    }
  }'
```

### 3.3. Available Model Keys

| Stage | Example Models |
|-------|----------------|
| **inference** | `openai:gpt-4o-mini`, `gemini:gemini-2.5-flash`, `openai:gpt-4o` |
| **rewrite** | `openai:gpt-4o-mini`, `gemini:gemini-2.5-flash` |
| **rerank** | `openai:gpt-4o-mini`, `openai:gpt-4o` |
| **summary** | `openai:gpt-4o-mini`, `gemini:gemini-2.5-flash` |
| **tools_synth** | `openai:gpt-4o-mini`, `gemini:gemini-2.5-flash` |

---

## 4. Token Accounting & Namespaces

### 4.1. Namespace Patterns

The system uses different namespace patterns for proper token accounting isolation:

| Approach | Namespace Pattern | Source | Example |
|----------|------------------|--------|---------|
| **Stateless** | `user_id:conversation_id` | Request params | `user123:conv456` |
| **Session-Based** | `session:{session_id}` | Auto-generated | `session:12d8cd79-...` |

### 4.2. Token Accounting Examples

#### Stateless Token Accounting
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG?",
    "history": [],
    "params": {
      "user_id": "user123",
      "conversation_id": "conv456",
      "top_k": 5
    }
  }'
```
- **Namespace:** `user123:conv456`
- **Token tracking:** Isolated per conversation
- **Use case:** Web frontend with multiple conversations per user

#### Session-Based Token Accounting
```bash
curl -X POST http://localhost:8000/chat/12d8cd79-0ee8-4dcd-97a5-5983effcbccd \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG?",
    "history": [],
    "params": {
      "user_id": "user123",  # Optional, for analytics
      "top_k": 5
    }
  }'
```
- **Namespace:** `session:12d8cd79-0ee8-4dcd-97a5-5983effcbccd`
- **Token tracking:** Isolated per session
- **Use case:** Mobile app with session-based conversations

### 4.3. Token Accounting Response

Both APIs return token usage metrics in the response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "metrics": {
    "tokens": {
      "embedding": 1500,
      "rewrite": 120,
      "rerank": 80,
      "inference": 250
    },
    "cost": {
      "embedding": 0.003,
      "rewrite": 0.002,
      "rerank": 0.001,
      "inference": 0.005
    }
  },
  "turn_metrics": {
    "tokens": {"total": 1950},
    "cost": {"total": 0.011}
  },
  "conversation_totals": {
    "tokens": {"total": 5000},
    "cost": {"total": 0.025},
    "messages": 3
  }
}
```

### 4.4. Benefits of Namespace Isolation

- **Cost Tracking** - Monitor tokens per user/conversation/session
- **Cache Management** - Separate caches for different contexts
- **Resource Isolation** - Prevent cross-contamination of data
- **Usage Analytics** - Track patterns per namespace
- **Multi-tenant Safety** - Ensure data isolation between users

---

## 4. Session vs Stateless Comparison

### When to Use Session-Based API

| Scenario | Recommended API | Reason |
|----------|----------------|--------|
| **Web frontend** | Stateless (`/chat`) | Simpler, client-managed state |
| **Mobile apps** | Session-based (`/chat/{session_id}`) | Server-side persistence |
| **Backend integrations** | Session-based | Automatic context management |
| **Multi-device access** | Session-based | Shared conversation state |
| **Simple API calls** | Stateless | No session setup needed |
| **Long-running conversations** | Session-based | Automatic history management |

### Quality and Performance

Both APIs provide:
- **Identical RAG pipeline quality**
- **Same retrieval and reranking**
- **Same query rewrite logic**
- **Same LLM inference models**
- **Same tool execution capabilities**

The only difference is **history management**:
- **Stateless:** Client sends full history each request
- **Session-based:** Server maintains and manages history

### Session Context Management

The session manager automatically:
- **Maintains conversation history** across requests
- **Applies token limits** to prevent context overflow
- **Preserves conversation flow** for follow-up questions
- **Handles context truncation** when token limits are exceeded

**Context Building Logic:**
```python
# From ChatSessionManager.get_context()
for msg in reversed(messages):
    msg_tokens = len(msg["content"].split())
    if total_tokens + msg_tokens <= max_history_tokens:
        context.append(msg)
    else:
        break
```

---
