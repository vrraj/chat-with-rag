# SSE (Server-Sent Events) Architecture for Chat-with-RAG

> **About this document**
>
> This page explains the **Server-Sent Events (SSE) architecture** for the *Chat-with-RAG* system, including real-time streaming implementation, event formats, and integration details.
>
> **Note:** If you landed here directly (for example from documentation hosting or search), start with the repository **[README](https://github.com/vrraj/chat-with-rag/blob/main/README.md)** to see how to run the system locally and try the interactive demo.

## 1. Overview

This project implements Server-Sent Events (SSE) to enable real-time streaming of different stages in the chat-with-RAG pipeline. SSE allows the server to push updates to the client over a single HTTP connection, providing a continuous stream of JSON-encoded events representing the progress and results of chat processing stages. This approach enhances user experience by delivering incremental updates without requiring the client to poll the server repeatedly.

## 2. Key Modules and Their Purpose

- **`stream_emit.py`**  
  Responsible for emitting SSE events. It formats messages according to the SSE protocol and sends them to connected clients.

- **`stream_registry.py`**  
  Manages the registry of active SSE consumers. It tracks which clients are subscribed to which query IDs and handles registration and deregistration.

- **SSE Endpoint in `stream_stages.py`**  
  Implements the HTTP endpoint that clients connect to for receiving SSE streams. The endpoint is mounted at `/chat/stream/stages` and ties together the registry and emitter to provide live updates of pipeline stages.

- **`chat_manager.py`**  
  Coordinates the overall chat processing logic, triggering events at various stages and interacting with the SSE system to stream updates.

- **Frontend `chat.js`**  
  Client-side JavaScript that establishes the SSE connection, listens for incoming events, and updates the user interface accordingly.

## 3. Event Structure

Events sent over SSE follow a JSON schema with different types depending on their purpose:

- **Stage Event**  
  Represents a progress update for a particular stage in the pipeline.
  ```json
  {
    "type": "stage",
    "stage": "Query Rewrite",
    "status": "completed",
    "data": { /* stage-specific data */ }
  }
  ```

- **Debug Event**  
  Contains debug information useful for development or troubleshooting.
  ```json
  {
    "type": "debug",
    "stage": "_connected",
    "id": "query_id_here",
    "note": "sse-connected"
  }
  ```

- **Keepalive Event**  
  Sent periodically to keep the connection alive and prevent timeouts.
  ```json
  {
    "type": "keepalive",
    "id": "query_id_here"
  }
  ```

- **Final Answer Event**  
  Contains the final chat response.
  ```json
  {
    "type": "stage",
    "stage": "Final Answer",
    "final": true,
    "finalContent": "The actual answer text",
    "finalHtml": "<p>Formatted answer</p>"
  }
  ```

## 4. Lifecycle of an SSE Stream

1. **Client Connects:** The frontend establishes an SSE connection to the server's SSE endpoint at `/chat/stream/stages` with a unique `query_id`.

2. **Registration:** The server registers the client in `stream_registry.py` to track the subscription.

3. **Event Emission:** As the chat pipeline progresses, `chat_manager.py` emits events via `stream_emit.py` to the registered client. Events are placed in a queue associated with the `query_id`.

4. **Queue Pre-drain:** When a client connects, any events already in the queue are sent first to provide immediate context.

5. **Keepalive Messages:** If no events are received for 12 seconds, a keepalive event is automatically sent to maintain the connection and prevent timeouts.

6. **Client Disconnects:** When the client closes the connection or navigates away, the server detects disconnection via `request.is_disconnected()` and deregisters the client.

7. **Server Cleanup:** The registry ensures no stale consumers remain, preventing resource leaks. A close sentinel (`None`) can be sent to explicitly end the stream.

## 5. Testing from CLI

You can test the SSE endpoint using `curl`:

```bash
curl -N http://localhost:8000/chat/stream/stages?query_id=12345
```

The `-N` flag disables buffering to stream events as they arrive.

Alternatively, use Python to test and time events:

```python
import requests
import time

response = requests.get('http://localhost:8000/chat/stream/stages?query_id=12345', stream=True)

start = time.time()
for line in response.iter_lines():
    if line:
        print(line.decode())
    if time.time() - start > 10:  # Stop after 10 seconds
        break
```

## 6. How to Verify Server Cleanup

Inspect the `stream_registry.py` to check active consumers. The registry maintains a mapping of `query_id` to connected clients. After clients disconnect, their entries should be removed promptly to avoid memory leaks.

You can add debug logs or expose an admin endpoint to report current registry state for verification.

## 7. Notes on Keepalives and Connection Management

- **Keepalives:** Sent automatically every 12 seconds when no other events are present. These events carry the `query_id` and are of type `keepalive` to prevent client or proxy timeouts.

- **Queue Pre-drain:** When a client connects, any events already queued for that `query_id` are sent immediately. This ensures the client sees relevant context even if it connected after processing started.

- **Connection Detection:** The server continuously checks `request.is_disconnected()` to detect when clients have disconnected and cleanup resources promptly.

- **Close Sentinel:** A `None` payload can be sent to explicitly close the stream. This is used for graceful shutdown of specific streams.

## 8. Future Extension: Optional Answer Streaming

A planned enhancement is to make **answer streaming configurable** based on use case:

- **Streaming:** Better perceived responsiveness for interactive use
- **Non-streaming:** Faster total response time for batch/API use cases

Implementation would include a configurable option (`stream_answer: true/false`) to let users choose the optimal approach for their needs.

## 9. Best Practices

- **Query ID Scope:** Each request generates a unique `query_id` for streaming. Multiple concurrent requests from the same user will have different `query_id`s.

- **Conversation ID:** Separate from `query_id`, the `conversation_id` persists across the entire conversation session for state management.

- **Stateful vs Stateless:** 
  - **Stateless** (`/chat`): Uses `query_id` + `conversation_id` 
  - **Stateful** (`/chat/{session_id}`): Uses `session_id` for server-managed conversation state

- **Connection Cleanup:** Frontend automatically closes SSE connections and clears server-side summaries on page unload.

---

# TL;DR

This SSE architecture streams real-time chat pipeline stages using a set of coordinated modules (`stream_emit.py`, `stream_registry.py`, `stream_stages.py`, `chat_manager.py`, and `chat.js`). Events follow a JSON schema for stages, keepalives, and debug info. Clients connect via SSE endpoints, receive incremental updates, and disconnect gracefully. Testing can be done via `curl` or Python scripts. Proper registry management ensures resource cleanup. Future plans include streaming final model answers with delta events. Follow best practices for consumer management and connection lifecycle.
