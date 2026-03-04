# SSE (Server-Sent Events) Architecture for Chat-with-RAG

## 1. Overview

This project implements Server-Sent Events (SSE) to enable real-time streaming of different stages in the chat-with-RAG pipeline. SSE allows the server to push updates to the client over a single HTTP connection, providing a continuous stream of JSON-encoded events representing the progress and results of chat processing stages. This approach enhances user experience by delivering incremental updates without requiring the client to poll the server repeatedly.

## 2. Key Modules and Their Purpose

- **`stream_emit.py`**  
  Responsible for emitting SSE events. It formats messages according to the SSE protocol and sends them to connected clients.

- **`stream_registry.py`**  
  Manages the registry of active SSE consumers. It tracks which clients are subscribed to which query IDs and handles registration and deregistration.

- **SSE Endpoint in `stream_stages.py`**  
  Implements the HTTP endpoint that clients connect to for receiving SSE streams. It ties together the registry and emitter to provide live updates of pipeline stages.

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
    "stage": "embedding",
    "status": "completed",
    "data": { /* stage-specific data */ }
  }
  ```

- **Keepalive Event**  
  Sent periodically to keep the connection alive and prevent timeouts.
  ```json
  {
    "type": "keepalive"
  }
  ```

- **Debug Event**  
  Contains debug information useful for development or troubleshooting.
  ```json
  {
    "type": "debug",
    "message": "Debug information here"
  }
  ```

## 4. Lifecycle of an SSE Stream

1. **Client Connects:** The frontend establishes an SSE connection to the server's SSE endpoint with a unique `query_id`.

2. **Registration:** The server registers the client in `stream_registry.py` to track the subscription.

3. **Event Emission:** As the chat pipeline progresses, `chat_manager.py` emits events via `stream_emit.py` to the registered client.

4. **Keepalive Messages:** Periodic keepalive events are sent to maintain the connection.

5. **Client Disconnects:** When the client closes the connection or navigates away, the server deregisters the client, cleaning up resources.

6. **Server Cleanup:** The registry ensures no stale consumers remain, preventing resource leaks.

## 5. Testing from CLI

You can test the SSE endpoint using `curl`:

```bash
curl -N http://localhost:8000/sse?query_id=12345
```

The `-N` flag disables buffering to stream events as they arrive.

Alternatively, use Python to test and time events:

```python
import requests
import time

response = requests.get('http://localhost:8000/sse?query_id=12345', stream=True)

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

## 7. Notes on Keepalives and Normal “Unregistered Consumer Loop” Messages

- **Keepalives:** Sent periodically to prevent client or proxy timeouts. These events carry minimal data and are of type `keepalive`.

- **Unregistered Consumer Loop Messages:** If a consumer is no longer registered (e.g., client disconnected), the server may send a message indicating this state before closing the stream. This informs clients about disconnection reasons.

## 8. Future Extension: Streaming the Final Model Answer

Planned enhancements include streaming the final model answer in real-time using delta events:

- **Delta Events:** Incremental partial answers streamed as they are generated.

- **Answer-Final Event:** Marks the completion of the final answer.

This will improve responsiveness by allowing users to see model outputs as they are produced.

## 9. Best Practices

- **One Consumer per `query_id`:** Ensure only one client consumes events per `query_id` to avoid duplicate processing.

- **Proper Cleanup:** Always deregister consumers on disconnect to free resources.

- **Browser Side Close:** Implement logic on the frontend to close SSE connections when no longer needed (e.g., on page unload).

---

# TL;DR

This SSE architecture streams real-time chat pipeline stages using a set of coordinated modules (`stream_emit.py`, `stream_registry.py`, `stream_stages.py`, `chat_manager.py`, and `chat.js`). Events follow a JSON schema for stages, keepalives, and debug info. Clients connect via SSE endpoints, receive incremental updates, and disconnect gracefully. Testing can be done via `curl` or Python scripts. Proper registry management ensures resource cleanup. Future plans include streaming final model answers with delta events. Follow best practices for consumer management and connection lifecycle.
