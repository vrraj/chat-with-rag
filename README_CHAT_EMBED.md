# Embedded Chat UI (`chat-embed.html`)

This document describes the **embeddable chat UI** built on top of the existing stateless `/chat` endpoint.

It is focused on:

- `frontend/chat-embed.html` – minimal chat surface intended for iframes and widgets.
- `frontend/static/chat-embed.js` – the JS client that talks to `/chat`.
- `frontend/static/embed-loader.js` – a small helper script for third-party sites.

The backend logic (including `backend/chat/chat_manager.py` and `POST /chat` semantics) is unchanged and shared with the main `frontend/chat.html` app. For the low-level API details, see:

👉 **[README_CHAT_API.md](README_CHAT_API.md)**

---

## 1. Overview

The embedded chat UI provides a **small, self-contained chat box** that can be dropped into other websites.

Key properties:

- **Embeddable** via `<iframe>` or a one-line `<script>` tag.
- **No parameter sidebar or metrics bar** – those controls are hidden.
- **Configuration via URL query parameters**.
- Reuses the same `/chat` endpoint and `params` contract as documented in `README_CHAT_API.md`.

The intent is to allow a host site to preconfigure behavior (retrieval, rewrite, tools, etc.) and expose a simple “chat with my docs/data” experience.

---

## 2. Files

- **`frontend/chat-embed.html`**
  - Minimal HTML shell for the embeddable chat UI.
  - Includes `static/chat.css` for styling (same base chat bubbles as `chat.html`).
  - Includes `static/chat-embed.js` for behavior.

- **`frontend/static/chat-embed.js`**
  - Lightweight client that:
    - Parses query parameters from the URL.
    - Builds `params` for `/chat`.
    - Renders chat bubbles and sends messages.

- **`frontend/static/embed-loader.js`**
  - Helper that host sites can load with a `<script>` tag.
  - Automatically injects an `<iframe>` pointing at `chat-embed.html` with the configured query parameters.

Backend files are **not modified** and are documented separately in `README_CHAT_API.md` and `TECHNICAL_OVERVIEW.md`.

---

## 3. Runtime behavior

### 3.1 `chat-embed.html`

- Renders:
  - `#embed_chat_history` – scrollable history region.
  - `#embed_chat_input` – textarea for user input.
  - `#embed_send_button` – send button.
  - A small footer for attribution.
- Does **not** render:
  - The left-hand parameter sidebar.
  - The metrics bar.
  - The models modal.

All behavior is driven by `static/chat-embed.js`.

### 3.2 `chat-embed.js`

`chat-embed.js` is responsible for:

- Parsing config from the current URL (`window.location.search`).
- Maintaining a **conversation id** (either provided by caller or locally generated).
- Building a `ChatRequest` payload and calling `POST /chat`.
- Rendering user and assistant messages in the embed UI.

It uses the exact same request shape as `README_CHAT_API.md` describes:

```jsonc
{
  "message": "<user text>",
  "use_web_search": false,
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "params": { /* see below */ }
}
```

---

## 4. Configuration via URL query parameters

The embed page is configured by query parameters on the `chat-embed.html` URL. All fields are **optional**. If a field is omitted, the backend defaults apply (via `Settings` / `run_pipeline`).

### 4.1 Retrieval

- `top_k` – integer or float-like string.
- `score_threshold` – float between 0 and 1.

### 4.2 Summarizer / history

- `chat_history_window_turns`
- `raw_tail_turns`
- `summarizer_max_input_tokens`
- `summarizer_max_output_tokens`

All should be integers if provided.

### 4.3 Inference

- `temperature` – float.
- `top_p` – float.
- `max_output_tokens` – integer.

### 4.4 Query rewrite

- `enable_query_rewrite` – `true` / `false` / `1` / `0` / `yes` / `no`.
- `rewrite_confidence_threshold` – float.
- `rewrite_tail_turns` – integer.

### 4.5 Tools

- `use_tools` – `true` / `false` / `1` / `0` / `yes` / `no`.

### 4.6 Provider/model overrides (optional)

These map directly to the stage-spec overrides used in `resolve_stage_specs`:

- `inference_provider`, `inference_model`
- `rewrite_provider`, `rewrite_model`
- `summary_provider`, `summary_model`
- `rerank_provider`, `rerank_model`

If provided, they are passed through as strings to the backend and interpreted there.

### 4.7 UX / observability

- `show_processing_steps`
  - Boolean-like string; defaults to `false` in the embed client.
  - When `true`, intermediate SSE processing steps will still be emitted by the backend, but `chat-embed.js` does not currently visualize them.

- `conversation_id`
  - Explicit conversation identifier to use.
  - Useful if the embedding site wants deterministic IDs.

- `namespace`
  - Alternative way to provide a conversation identifier.
  - If `conversation_id` is not provided, `namespace` will be used for `params.conversation_id`.

- `mode`
  - Optional string tag, defaults to `embed`.
  - Sent as `params.mode` for logging / analytics.

### 4.8 Internal fields added by the client

`chat-embed.js` always adds:

- `query_id` – an 8-character ID per turn, generated in the browser.
- `conversation_id` – chosen using this logic:

  1. If `conversation_id` query param is present → use it.
  2. Else if `namespace` query param is present → use it.
  3. Else → use `sessionStorage['conversation_id_embed']` if set.
  4. Else → generate a new 8-character ID and store it in `sessionStorage` under `conversation_id_embed`.

These align with the `params` contract in **`README_CHAT_API.md`**.

---

## 5. Example embed URLs

### 5.1 Minimal default embed

```text
/chat-embed.html
```

Relies entirely on backend defaults; uses a generated `conversation_id`.

### 5.2 Preset retrieval + inference

```text
/chat-embed.html?top_k=8&score_threshold=0.35&temperature=0.4&max_output_tokens=300
```

### 5.3 With query rewrite and tools disabled

```text
/chat-embed.html?
  top_k=8&
  score_threshold=0.35&
  temperature=0.4&
  max_output_tokens=300&
  enable_query_rewrite=true&
  rewrite_confidence_threshold=0.67&
  rewrite_tail_turns=1&
  use_tools=false
```

### 5.4 Explicit conversation / namespace

```text
/chat-embed.html?namespace=docs-help&top_k=5
```

This will use `params.conversation_id = "docs-help"` for all turns in that iframe.

---

## 6. Using `embed-loader.js` on a host site

For third-party websites that can only add a `<script>` tag, `embed-loader.js` provides a simple integration path.

### 6.1 Basic usage

```html
<div id="support-chat"></div>
<script
  src="https://your-app.com/static/embed-loader.js"
  data-target="#support-chat"
  data-top_k="8"
  data-score_threshold="0.35"
  data-temperature="0.4"
  data-max_output_tokens="300"
  data-enable_query_rewrite="true"
  data-use_tools="false"
  data-namespace="docs-help"
  data-width="100%"
  data-height="450px"
></script>
```

`embed-loader.js` will:

1. Read its own `data-*` attributes via `script.dataset`.
2. Use `data-target` as a CSS selector to find the host element.
3. Treat all other `data-*` keys (except `target`, `width`, `height`) as query parameters to `chat-embed.html`.
4. Compute the `chat-embed.html` URL **relative to the script URL**.
5. Inject an `<iframe>` inside the target element with:
   - `src = computed chat-embed.html URL + querystring`.
   - `style.width = data-width` (default: `100%`).
   - `style.height = data-height` (default: `400px`).
   - No border.

### 6.2 Notes for integrators

- The host page must allow loading the app’s domain in an iframe.
- If you change the path to `chat-embed.html`, update the relative URL logic in `embed-loader.js` accordingly.
- All configuration parameters are the same as the query parameters described in [Section 4](#4-configuration-via-url-query-parameters).

---

## 7. Testing locally

1. **Start the stack** (see `README.md`):

   ```bash
   make start
   ```

2. **Open the embed page directly**:

   ```text
   http://localhost:8000/chat-embed.html
   ```

3. **Try with custom params**:

   ```text
   http://localhost:8000/chat-embed.html?top_k=8&score_threshold=0.35&temperature=0.4
   ```

4. Optionally, create a small HTML page that includes `static/embed-loader.js` and serves it from the same domain to simulate third-party embedding.

---

## 8. Relationship to the main chat UI

- Both `chat.html` and `chat-embed.html` send requests to **the same** `POST /chat` endpoint.
- Both use the same `params` keys as defined in **`README_CHAT_API.md`**.
- `chat-embed.html` is intentionally **minimal** and is suitable for iframes and third-party widgets.
- Any future changes to the backend `/chat` contract should be reflected in **both** `README_CHAT_API.md` and this document, to keep embed integrators aligned.
