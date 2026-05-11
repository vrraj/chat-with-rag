# Chat User Guide

## High-Level Overview

The chat system in `backend/chat/chat_manager.py` provides a comprehensive conversational AI interface with RAG (Retrieval-Augmented Generation) capabilities. It supports both stateless and stateful chat modes, multi-stage processing pipelines, tool execution, and detailed metrics tracking.

### Key Concepts

Before diving into the details, here are the core concepts you need to understand:

1. **Domains**: Multi-tenant support that routes requests to different data collections and prompt templates
2. **Stateless vs Stateful**: Two ways to interact - per-request (stateless) or in-memory conversation (stateful)
3. **Pipeline Stages**: Multi-step processing from query rewrite to final answer generation
4. **Tools**: External capabilities (web search, stock prices, etc.) that the LLM can call
5. **Metrics**: Detailed token and cost tracking per stage and per conversation
6. **Namespaces**: Scoping mechanism for metrics and caches per conversation/user
7. **Caching**: Summary and history caching to reduce costs and improve performance

## Domains

### What are Domains?

Domains enable multi-tenancy by allowing different "spaces" within the same application. Each domain can have:
- Its own Qdrant collection (separate document index)
- Its own embedding model
- Its own prompt templates (system instructions)

### When to Use Domains

Use domains when you need:
- **Multi-tenant applications**: Separate data for different customers or departments
- **Specialized assistants**: Different prompt behaviors for different use cases (e.g., finance vs. general)
- **Data isolation**: Keep document collections separate

### Domain Configuration

Domains are configured in `settings.DOMAIN_EMBEDDING_CONFIG`:

```python
DOMAIN_EMBEDDING_CONFIG = {
    "finance": {
        "collection_name": "finance_collection",
        "embedding_model_key": "openai:text-embedding-3-small"
    },
    "legal": {
        "collection_name": "legal_collection",
        "embedding_model_key": "openai:text-embedding-3-large"
    },
    "default": {
        "collection_name": "documents",
        "embedding_model_key": "openai:text-embedding-3-small"
    }
}
```

### Domain Resolution Priority

When a request comes in, the system determines which domain to use in this order:

1. `params.active_domain` - Request-specific domain for data retrieval
2. `params.prompt_domain` - Request-specific domain for prompts
3. `settings.active_domain` - Default domain from configuration
4. "default" - Fallback to default domain

### Active Domain vs Prompt Domain

- **active_domain**: Controls which Qdrant collection and embedding model to use for retrieval
- **prompt_domain**: Controls which prompt templates (system instructions) to use for LLM stages

These can be different - you might retrieve from a finance collection but use general prompts.

### Example Usage

```python
# Request to finance domain
payload = {
    "message": "What's Apple's stock price?",
    "params": {
        "active_domain": "finance",  # Use finance collection
        "prompt_domain": "finance"    # Use finance prompts
    }
}

# Request with mixed domains
payload = {
    "message": "Tell me about contracts",
    "params": {
        "active_domain": "legal",     # Use legal collection
        "prompt_domain": "default"    # Use general prompts
    }
}
```

## Entry Points

### 1. handle_chat (Stateless Entry Point)

**Location**: `backend/chat/chat_manager.py:4182`

The `handle_chat` function is the main stateless entry point for chat requests. It processes each request independently without maintaining conversation state in memory.

**Function Signature**:
```python
def handle_chat(payload: Dict[str, Any]) -> Dict[str, Any]
```

**Input Payload Structure**:
```python
{
    "message": str,              # User query
    "history": List[Dict],       # Conversation history (optional)
    "params": {
        "top_k": int,            # Number of documents to retrieve (default: 8)
        "score_threshold": float, # Minimum similarity score (default: 0.0)
        "enable_query_rewrite": bool,  # Enable query rewriting (default: False)
        "use_tools": bool,        # Enable tool execution (default: False)
        "use_web_search": bool,   # Enable web search (default: False)
        "show_sources": bool,     # Display sources in response (default: True)
        "show_processing_steps": bool,  # Show intermediate stages (default: True)
        "render_html": bool,      # Render markdown to HTML (default: False)
        "active_domain": str,      # Domain for routing (default: from settings)
        "prompt_domain": str,      # Domain for prompts (default: from settings)
        "conversation_id": str,   # Conversation ID for namespace scoping
        "user_id": str,           # User ID for namespace scoping
        "session_id": str,        # Session ID for namespace scoping
        "query_id": str,          # Request ID for SSE tracking
        "model_keys": {           # Optional model registry keys per stage
            "inference": str,
            "rewrite": str,
            "summary": str,
            "rerank": str,
            "embedding": str
        },
        "rewrite_tail_turns": int,        # Turns to keep verbatim for rewrite (default: 1)
        "rewrite_summary_turns": int,     # Turns to summarize for rewrite (default: 3)
        "rewrite_confidence_threshold": float  # Rewrite acceptance threshold (default: 0.6)
    }
}
```

**Output Response Structure**:
```python
{
    "answer": str,              # Final answer text
    "response": str,            # Alias for answer (legacy compatibility)
    "answer_html": str,         # Rendered HTML (if render_html=True)
    "sources": List[Dict],      # Source documents used
    "metrics": {
        "vectors_retrieved": int
    },
    "turn_metrics": Dict,       # Per-turn metrics
    "conversation_totals": Dict, # Conversation-level metrics
    "tools_used": List[str],    # Tools executed
    "rewrite_display": Dict     # Query rewrite information
}
```

### 2. ChatManager.chat (Stateful Entry Point)

**Location**: `backend/chat/chat_manager.py:1985`

The `ChatManager.chat` method provides a stateful chat interface that maintains conversation history in memory.

**Function Signature**:
```python
def chat(self, message: str, context: List[Dict], use_web_search: bool | None = None, params: Dict[str, Any] | None = None) -> Dict
```

**Key Differences from handle_chat**:
- Maintains `self.chat_history` in memory
- Supports session-based chunked history management
- Delegates to the same orchestrator pipeline as handle_chat
- Uses session_id for namespace-based token accounting

## Namespaces

### What are Namespaces?

Namespaces provide scoping for metrics and caches across conversations. They ensure that:
- Token counts and costs are tracked per conversation
- Summary caches are isolated per user/conversation
- Different conversations don't interfere with each other

### Namespace Format

Namespaces are constructed from user and conversation identifiers:
- Format: `{user_id}:{conversation_id}` or `session:{session_id}`
- Example: "user:123:convo:456" or "session:abc-123-def"

### When Namespaces Matter

Namespaces are important when:
- You have multiple concurrent conversations
- You need accurate per-conversation cost tracking
- You want to prevent cache pollution between conversations

### Providing Namespace Information

To enable namespace scoping, include these in your request:

```python
params = {
    "user_id": "user-123",           # User identifier
    "conversation_id": "conv-456",   # Conversation identifier
    # OR
    "session_id": "session-abc"      # Session identifier
}
```

If no namespace is provided, the system uses a global accumulator (not recommended for production multi-user scenarios).

## How It Works: The Processing Pipeline

### Overview Flow

When you send a message to the chat system, it goes through these stages:

```
User Message
    ↓
[Optional] Query Rewrite (improve ambiguous queries)
    ↓
Context Retrieval (search vector database)
    ↓
[Optional] Reranking (reorder results for relevance)
    ↓
[Optional] Context Summarization (fit within token limits)
    ↓
Prompt Construction (build LLM prompt)
    ↓
LLM Inference (generate response or tool calls)
    ↓
[Optional] Tool Execution (call external APIs)
    ↓
[Optional] Tools Synthesis (combine tool results)
    ↓
Final Answer
```

### Which Stages Run?

Not all stages run on every request:

- **Query Rewrite**: Only if enabled + history exists + heuristic passes
- **Reranking**: Skipped if results are few or already high-quality
- **Context Summarization**: Only if context is too long
- **Tool Execution**: Only if tools are enabled + LLM requests tools
- **Tools Synthesis**: Only if tools were executed

### Domain Routing in the Pipeline

Domain resolution happens early in the pipeline (before context retrieval):
1. System resolves `active_domain` from params/settings
2. Creates a fresh QdrantDB instance with domain-specific collection
3. Uses domain-specific embedding model for the query
4. All retrieval happens in the domain-specific collection

Prompt domain is resolved separately for each LLM stage (rewrite, inference, rerank, summary, synthesis).

## Pipeline Stages

The chat system processes requests through a multi-stage pipeline orchestrated by `run_pipeline`:

### Stage 1: Query Rewrite (Optional)

**Purpose**: Rewrites ambiguous or underspecified queries using conversation context.

**Trigger Conditions**:
- `enable_query_rewrite` is True
- History exists (not the first turn)
- Heuristic check passes (short queries, coreferences)

**Parameters**:
- `rewrite_tail_turns`: Number of recent turns to keep verbatim (default: 1)
- `rewrite_summary_turns`: Number of older turns to summarize (default: 3)
- `rewrite_confidence_threshold`: Minimum confidence to accept rewrite (default: 0.6)

**Process**:
1. Split history into tail (verbatim) and summary sections
2. Summarize older turns if `rewrite_summary_turns > 0`
3. Call rewrite LLM with tail + summary + current message
4. Accept rewrite if confidence >= threshold
5. Otherwise, use original query

**Metrics Tracked**:
- Input/output tokens for rewrite stage
- Cost of rewrite operation
- Whether rewrite was applied and reason

### Stage 2: Context Retrieval

**Purpose**: Retrieves relevant documents from Qdrant vector database.

**Parameters**:
- `top_k`: Number of documents to retrieve (default: 8)
- `score_threshold`: Minimum similarity score (default: 0.0)
- `exact_match`: Use exact matching vs HNSW (default: False)

**Process**:
1. Embed the query using the configured embedding model
2. Search Qdrant for similar vectors
3. Return documents with scores above threshold

**Domain-Aware Routing**:
- Resolves `active_domain` from params or settings
- Routes to domain-specific Qdrant collection
- Uses domain-specific embedding model

**Metrics Tracked**:
- Input tokens for embedding
- Cost of embedding operation
- Number of vectors retrieved

### Stage 3: Reranking (Conditional)

**Purpose**: Reorders retrieved documents to improve relevance.

**Trigger Conditions** (skip if any true):
- Fewer than 2 results
- Fewer than `re_ranker_input_rows` results (default: 5)
- Exact match found in top 5 with score >= `rerank_exact_match_min_score` (default: 0.80)
- Top result is a clear winner (score >= 0.65, margin >= 0.15)

**Parameters**:
- `re_ranker_input_rows`: Max candidates to rerank (default: 5)
- `rerank_clear_winner_min_top1`: Min score for clear winner (default: 0.65)
- `rerank_clear_winner_min_delta`: Min margin for clear winner (default: 0.15)
- `rerank_exact_match_min_score`: Min score for exact match (default: 0.80)
- `reranker_chunk_size`: Characters per candidate snippet (default: 600)

**Process**:
1. Extract text snippets from candidates
2. Build rerank prompt with query and candidates
3. Call rerank LLM to rank candidates
4. Reorder results based on LLM output

**Metrics Tracked**:
- Input/output tokens for rerank
- Cost of rerank operation
- Number of candidates reranked

### Stage 4: Context Summarization (Conditional)

**Purpose**: Summarizes retrieved context to fit within token limits.

**Trigger Conditions**:
- Context length exceeds summarizer limits
- Summarization is enabled

**Parameters**:
- `summarizer_temperature`: Temperature for summarization (default: 0.3)
- `summarizer_max_input_tokens`: Max input tokens (default: 512)
- `summarizer_max_output_tokens`: Max output tokens (default: 128)

**Process**:
1. Check cache for existing summary
2. If cache miss, generate summary using LLM
3. Cache the result with namespace key

**Caching**:
- Namespace-scoped cache keys (e.g., "user:123|convo:456|summary")
- Idle eviction after `summary_cache_idle_ttl_seconds` (default: 3600)

**Metrics Tracked**:
- Input/output tokens for summarization
- Cost of summarization
- Cache hit/miss status

### Stage 5: Prompt Construction

**Purpose**: Builds the final prompt for the inference LLM.

**Components**:
- System prompt (from prompt registry)
- Conversation summary (if applicable)
- Recent conversation history
- Retrieved context
- User query

**Prompt Registry**:
- Domain-specific prompts via `prompt_domain`
- Registry path: `inference_prompt_registry_path` (default: "")
- Templates support variable interpolation

**Style Options**:
- `messages`: Structured message format (system/user/assistant roles)
- `flat`: Single string format

### Stage 6: LLM Inference (Pass 1)

**Purpose**: Generates initial response or tool calls.

**Parameters**:
- `inference_provider`: LLM provider (default: "openai")
- `inference_model`: Model name or registry key
- `inference_temperature`: Temperature (default: 0.2)
- `inference_top_p`: Top-p sampling (default: 1.0)
- `max_inference_output_tokens`: Max output tokens (default: 800)
- `inference_reasoning_effort`: Reasoning effort level (default: "low")
- `debug_thoughts`: Include reasoning in output (default: True)

**Tool Calling** (if enabled):
- Tools from tool registry are included in the request
- LLM decides which tools to call
- Supports parallel tool calls

**Metrics Tracked**:
- Input tokens, cached tokens, output tokens
- Reasoning tokens (if applicable)
- Cost breakdown (prompt, cached, completion)
- Total cost

### Stage 7: Tool Execution (Conditional)

**Purpose**: Executes tools requested by the inference LLM.

**Process**:
1. Parse tool calls from LLM response
2. Execute each tool via registered executor
3. Collect tool outputs
4. Handle errors gracefully

**Tool Output Redaction**:
- Large artifacts (SVGs, images) are redacted for synthesis
- Compact metadata preserved for reference
- Artifacts re-injected after synthesis

**Metrics Tracked**:
- Tools executed and their outputs
- Tool execution time (if available)

### Stage 8: Tools Synthesis (Conditional)

**Purpose**: Synthesizes tool outputs into a final answer.

**Trigger**: Tools were executed in Stage 7

**Parameters**:
- Uses same provider/model as inference (by default)
- `tools_synth_max_output_tokens`: Max output tokens (default: same as inference)
- Does NOT include tool-calling params (no recursive tool calls)

**Process**:
1. Build synthesis prompt with:
   - Conversation summary
   - Recent history
   - Retrieved context
   - Tool outputs (redacted)
   - Original question
2. Call synthesis LLM
3. Inject registered artifacts (SVG, etc.) back into response
4. Handle cases where synthesis missed artifacts

**Metrics Tracked**:
- Input/output tokens for synthesis
- Cost of synthesis
- Separate from inference pass #1 metrics

## Parameter Settings

### Stage Configuration (resolve_stage_specs)

**Location**: `backend/chat/chat_manager.py:163`

The `resolve_stage_specs` function computes provider/model/kwargs for each pipeline stage.

**Output Structure**:
```python
{
    "embedding": {
        "provider": str,      # e.g., "openai", "gemini"
        "model": str,         # Embedding model name
        "kwargs": Dict
    },
    "rewrite": {
        "provider": str,
        "model": str,
        "kwargs": {
            "temperature": float,
            "max_output_tokens": int
        }
    },
    "summary": {
        "provider": str,
        "model": str,
        "kwargs": {
            "temperature": float,
            "max_output_tokens": int,
            "_max_input_tokens": int
        }
    },
    "rerank": {
        "provider": str,
        "model": str,
        "kwargs": {
            "temperature": float,
            "max_output_tokens": int
        }
    },
    "inference": {
        "provider": str,
        "model": str,
        "kwargs": {
            "temperature": float,
            "top_p": float,
            "max_output_tokens": int,
            "reasoning_effort": str,
            "debug_thoughts": bool,
            "tools": List  # if enable_tools
        }
    },
    "tools_synth": {
        "provider": str,
        "model": str,
        "kwargs": {
            "temperature": float,
            "max_output_tokens": int,
            "reasoning_effort": str,
            "debug_thoughts": bool
        }
    }
}
```

### Per-Request Overrides

Frontend can override settings via `params`:

**Model Overrides**:
- `inference_provider`, `inference_model`
- `rewrite_provider`, `rewrite_model`
- `summary_provider`, `summary_model`
- `rerank_provider`, `rerank_model`

**Model Registry Keys**:
- `model_keys.inference`: Registry key for inference stage
- `model_keys.rewrite`: Registry key for rewrite stage
- `model_keys.summary`: Registry key for summary stage
- `model_keys.rerank`: Registry key for rerank stage
- `model_keys.embedding`: Registry key for embedding stage

Registry keys take precedence over model names for accurate cost tracking.

### Configuration Settings

**Database Settings**:
- `qdrant_host`: Qdrant server host
- `qdrant_port`: Qdrant server port
- `collection_name`: Default Qdrant collection
- `embedding_model`: Embedding model provider key
- `embedding_model_key`: Embedding model registry key

**Retrieval Settings**:
- `top_k`: Default number of documents to retrieve (default: 8)
- `score_threshold`: Default similarity threshold (default: 0.0)
- `exact_match`: Use exact matching (default: False)

**Rewrite Settings**:
- `enable_query_rewrite`: Enable query rewriting (default: False)
- `rewrite_model`: Model for query rewrite
- `rewrite_temperature`: Temperature (default: 0.2)
- `rewrite_max_output_tokens`: Max output tokens (default: 128)
- `rewrite_tail_turns`: Turns to keep verbatim (default: 1)
- `rewrite_summary_turns`: Turns to summarize (default: 3)
- `rewrite_confidence_threshold`: Acceptance threshold (default: 0.6)

**Summarization Settings**:
- `summarizer_model`: Model for summarization
- `summarizer_temperature`: Temperature (default: 0.3)
- `summarizer_max_input_tokens`: Max input tokens (default: 512)
- `summarizer_max_output_tokens`: Max output tokens (default: 128)
- `summary_cache_idle_ttl_seconds`: Cache TTL (default: 3600)

**Rerank Settings**:
- `re_ranker_model`: Model for reranking
- `re_ranker_temperature`: Temperature (default: 0.0)
- `re_ranker_max_output_tokens`: Max output tokens (default: 64)
- `re_ranker_input_rows`: Max candidates to rerank (default: 5)
- `rerank_clear_winner_min_top1`: Min score for clear winner (default: 0.65)
- `rerank_clear_winner_min_delta`: Min margin for clear winner (default: 0.15)
- `rerank_exact_match_min_score`: Min score for exact match (default: 0.80)
- `reranker_chunk_size`: Characters per snippet (default: 600)

**Inference Settings**:
- `inference_provider`: LLM provider (default: "openai")
- `inference_model`: Model name or registry key
- `inference_model_key`: Model registry key
- `inference_temperature`: Temperature (default: 0.2)
- `inference_top_p`: Top-p sampling (default: 1.0)
- `max_inference_output_tokens`: Max output tokens (default: 800)
- `inference_reasoning_effort`: Reasoning effort (default: "low")
- `debug_thoughts`: Include reasoning (default: True)

**Tools Settings**:
- `enable_tools`: Enable tool execution (default: False)
- `tools_synth_max_output_tokens`: Synthesis max tokens (default: same as inference)
- `tool_registry_path`: Path to tool registry (default: "prompts/tool_registry.yaml")

**Web Search Settings**:
- `use_web_search`: Enable web search (default: False)

**Display Settings**:
- `display_sources_for_chat`: Show sources in chat mode (default: True)
- `display_sources_for_embed`: Show sources in embed mode (default: False)
- `show_processing_steps`: Show intermediate stages (default: True)

**Prompt Registry Settings**:
- `inference_prompt_registry_path`: Path to prompt registry (default: "")
- `prompt_domain_default`: Default domain for prompts (default: "")

**Domain Settings**:
- `active_domain`: Default active domain (default: "")
- `DOMAIN_EMBEDDING_CONFIG`: Domain-specific configuration

**History Settings**:
- `raw_tail_turns`: Turns to keep verbatim in history (default: 10)
- `chunk_manager_idle_ttl_seconds`: Chunk manager TTL (default: 3600)

**Debug Settings**:
- `debug_verbose`: Enable verbose debug logging (default: False)
- `debug_log_truncate_chars`: Truncate debug logs (default: 4000)

**Cost Settings**:
- `cost_basis_tokens`: Token basis for cost display (default: 1,000,000)

## Metrics Tracking

### Metrics Class

**Location**: `backend/chat/chat_manager.py:1126`

The `Metrics` class tracks per-turn and conversation-level metrics.

**Turn Metrics Structure**:
```python
{
    "embedding": {
        "model": str,
        "input_tokens": int,
        "cost": float
    },
    "rerank": {
        "model": str,
        "input_tokens": int,
        "output_tokens": int,
        "candidates_reranked": int,
        "cost": float
    },
    "summary": {
        "model": str,
        "applied": bool,
        "reason": str,
        "input_tokens": int,
        "output_tokens": int,
        "cost": float
    },
    "rewrite": {
        "model": str,
        "applied": bool,
        "reason": str,
        "input_tokens": int,
        "output_tokens": int,
        "cost": float
    },
    "inference": {
        "model": str,
        "input_tokens": int,
        "cached_tokens": int,
        "output_tokens": int,
        "reasoning_tokens": int,
        "cost_input": float,
        "cost_cached": float,
        "cost_output": float,
        "cost_total": float
    },
    "inference_tools_synth": {
        "model": str,
        "input_tokens": int,
        "cached_tokens": int,
        "output_tokens": int,
        "reasoning_tokens": int,
        "cost_input": float,
        "cost_cached": float,
        "cost_output": float,
        "cost_total": float
    },
    "totals": {
        "tokens": {
            "turn_total": int
        },
        "cost": {
            "turn_total": float
        }
    }
}
```

### Conversation Totals

**Structure**:
```python
{
    "tokens": {
        "embedding": int,
        "llm_input": int,      # prompt + cached tokens
        "llm_output": int,     # completion tokens
        "conversation_total": int
    },
    "costs": {
        "embedding": float,
        "llm_input": float,
        "llm_output": float,
        "total": float
    }
}
```

**Namespace Scoping**:
- Totals are scoped per namespace (e.g., "user:123|convo:456")
- Empty namespace uses default global accumulator
- Namespace derived from `user_id` + `conversation_id` or `session_id`

### Cost Calculation

Costs are computed via `_compute_stage_cost` using model registry pricing when available.

**Cost Components**:
- Prompt tokens (non-cached input)
- Cached tokens (discounted rate)
- Completion tokens (output)
- Reasoning tokens (special rate for reasoning models)

**Model Registry Keys**:
- Per-stage model keys from `params.model_keys` enable accurate cost lookup
- Fallback to model names if registry keys not provided

## Domain-Aware Routing

### Domain Resolution

**Priority Order**:
1. `params.active_domain`
2. `params.prompt_domain`
3. `settings.active_domain`
4. "default"

**Domain Configuration**:
```python
DOMAIN_EMBEDDING_CONFIG = {
    "finance": {
        "collection_name": "finance_collection",
        "embedding_model_key": "openai:text-embedding-3-small"
    },
    "default": {
        "collection_name": "default_collection",
        "embedding_model_key": "openai:text-embedding-3-small"
    }
}
```

### Per-Request Domain Routing

In `handle_chat`, a fresh QdrantDB instance is created per request with domain-specific settings:

```python
db = QdrantDB(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    collection_name=domain_collection,
    embedding_model_key=domain_embedding_model_key
)
```

### Prompt Domain

Separate from retrieval domain, controls prompt template selection:
- `params.prompt_domain` for per-request override
- `settings.prompt_domain_default` as fallback
- Used in prompt registry resolution for all LLM stages

## Caching Mechanisms

### Summary Cache

**Purpose**: Cache conversation summaries to avoid redundant summarization.

**Cache Keys**:
- Format: `{namespace}|{tag}`
- Example: "user:123|convo:456|summary" or "user:123|convo:456|rewrite"

**Cache Structure**:
- Module-level `_SUMMARY_CACHE` dict
- `_SUMMARY_NS_INDEX` for namespace-based clearing
- `_SUMMARY_NS_LAST_SEEN` for idle eviction

**Idle Eviction**:
- TTL: `summary_cache_idle_ttl_seconds` (default: 3600)
- Evicted automatically on cache access

**Cache Management Functions**:
- `clear_summaries_for_namespace(namespace)`: Clear all summaries for a namespace
- `_touch_namespace(namespace)`: Update last-seen timestamp
- `_evict_idle_namespaces()`: Evict idle entries

### Chunk Manager Cache

**Purpose**: Manage chunked history for long conversations.

**Cache Structure**:
- Module-level `_CHUNK_MANAGERS_BY_NS` dict
- `_CHUNK_MANAGERS_LAST_SEEN` for idle eviction

**Idle Eviction**:
- TTL: `chunk_manager_idle_ttl_seconds` (default: 3600)
- Evicted on chunk manager access

**Cache Management Functions**:
- `clear_chunk_manager_for_namespace(namespace)`: Clear chunk manager for namespace
- `_get_chunk_manager_for_namespace(namespace, settings)`: Get or create chunk manager

## Tool Execution Flow

### Tool Registry

**Location**: `prompts/tool_registry.yaml` (configurable via `tool_registry_path`)

**Registry Structure**:
```yaml
tools:
  - name: tool_name
    description: Tool description
    parameters:
      - name: param
        type: string
        description: Parameter description
    function:
      module: backend.tools.tool_module
      function: tool_function
    artifacts:
      - field: svg
        type: image/svg+xml
```

### Tool Execution Process

1. **Tool Discovery**:
   - `list_tools()` loads tools from registry
   - Filters out web_search if not explicitly requested

2. **Tool Calling**:
   - Inference LLM receives tool definitions
   - LLM decides which tools to call and with what parameters
   - Supports parallel tool calls

3. **Tool Execution**:
   - `get_executor(tool_name)` returns executor function
   - Executor called with tool parameters
   - Output collected in `tool_outputs_list`

4. **Output Redaction**:
   - `_redact_tool_outputs_for_synth()` removes large artifacts
   - Keeps compact metadata for synthesis
   - Artifacts re-injected after synthesis

5. **Tools Synthesis**:
   - Redacted outputs passed to synthesis LLM
   - Synthesis LLM generates final answer
   - Registered artifacts injected into response

### Tool Output Redaction

**Purpose**: Prevent large artifacts from overwhelming synthesis prompt.

**Redaction Rules**:
- Fields marked as artifacts in registry are redacted
- Replaced with placeholder: "Artifact payload omitted for synthesis. placeholder={name}"
- Non-artifact fields preserved verbatim

**Artifact Re-injection**:
- `_inject_registered_artifacts()` restores artifacts after synthesis
- Handles cases where synthesis missed/trimmed artifacts
- Preserves canonical SVG from tool output

## Error Handling

### LLM Rate Limits

When LLM providers hit rate limits, the system:
1. Detects `LLMError` with `kind="rate_limit"`
2. Returns user-friendly error message
3. Includes provider and model information
4. Closes SSE stream gracefully
5. Returns metrics collected so far

**Affected Stages**:
- Rewrite (rewrite model rate limit)
- Summarization (summarizer rate limit)
- Rerank (reranker rate limit)
- Inference (inference model rate limit)
- Tools Synthesis (synthesis model rate limit)

### Fallback Behavior

**Query Rewrite**:
- If rewrite fails, uses original query
- Logs error but continues pipeline

**Rerank**:
- If rerank fails, uses original order
- Logs error but continues pipeline

**Summarization**:
- If summarization fails, uses raw context
- Logs error but continues pipeline

**Tool Execution**:
- Individual tool failures logged
- Other tools continue execution
- Partial results passed to synthesis

## SSE Stage Emission

### Processing Stages

The system emits intermediate stages via SSE when `show_processing_steps` is True:

**Stages Emitted**:
1. "Query Rewrite" (if enabled)
2. "Rerank Retrieval Results" (if reranking)
3. "Skipping Rerank" (if rerank skipped)
4. "Generating Responses with Tools" (if tools enabled)
5. "Final Answer" (always)
6. "Done" (always, closes stream)

### Stage Content

**Final Answer Stage**:
- `finalContent`: Answer text
- `finalHtml`: Rendered HTML (if `render_html=True`)

**Other Stages**:
- No content, just stage name for progress indication

## History Management

### Chunked History

**Purpose**: Efficiently manage long conversations by chunking history.

**Chunk Size**:
- Configured via `raw_tail_turns` (default: 10)
- Number of recent turns kept verbatim

**Chunk Manager**:
- Per-session instance via `get_or_create_chunk_manager(session_id)`
- Maintains chunks of conversation history
- Provides efficient history access

### History Splitting

**Function**: `split_history_for_prompt(history, raw_tail, window_turns)`

**Parameters**:
- `raw_tail`: Turns to keep verbatim
- `window_turns`: Total turns to include (summarized + tail)

**Returns**:
- `to_sum`: Turns to summarize
- `tail`: Turns to keep verbatim

## Prompt Registry

### Registry Structure

**Location**: Configured via `inference_prompt_registry_path`

**Domain-Specific Prompts**:
```yaml
domains:
  finance:
    system_instruction: |
      You are a financial assistant...
    full_payload_template: |
      Context: {context}
      Question: {question}
```

### Prompt Resolution

**Functions**:
- `resolve_inference_prompt(registry_path, domain)`: Resolve inference prompt
- `resolve_rewrite_prompt(registry_path, domain)`: Resolve rewrite prompt
- `resolve_rerank_prompt(registry_path, domain)`: Resolve rerank prompt
- `resolve_summary_prompt(registry_path, domain)`: Resolve summary prompt
- `resolve_tools_synth_prompt(registry_path, domain)`: Resolve tools synthesis prompt

**Domain Selection**:
- `params.prompt_domain` takes precedence
- Falls back to `settings.prompt_domain_default`
- Finally falls back to default prompts

### Variable Interpolation

**Template Variables**:
- `{context}`: Retrieved context
- `{question}`: User question
- `{candidates_block}`: Rerank candidates
- `{summary}`: Conversation summary
- `{history}`: Conversation history

## Web Search Integration

### Web Search Client

**Location**: `backend/chat/web_search.py`

**Integration**:
- Enabled via `use_web_search` parameter
- Adds web search results to retrieved context
- Used as supplemental to vector search

**Process**:
1. Perform vector search first
2. Pass results to `get_web_context(query, existing_context)`
3. Web search client adds relevant web results
4. Combined context passed to inference

## Debug Logging

### Debug Helpers

**Function**: `_dbg(label, text)`

**Behavior**:
- Only logs when `settings.debug_verbose` is True
- Truncates output to `debug_log_truncate_chars` (default: 4000)
- Never breaks flow on logging errors

### Debug Information Logged

**Pipeline Stages**:
- Stage specs (provider, model, kwargs)
- Model keys from frontend
- Rewrite parameters
- Rerank decision logic
- Tool execution details

**Domain Routing**:
- Requested vs effective domain
- Collection and embedding model used

**Cache Operations**:
- Cache size before/after operations
- Cache hit/miss status

**Metrics**:
- Per-stage token counts
- Cost calculations
- Conversation totals

## Best Practices

### For Developers

1. **Use Namespace Scoping**: Always provide `user_id` and `conversation_id` to isolate metrics and caches per conversation.

2. **Model Registry Keys**: Use `model_keys` in params for accurate cost tracking rather than just model names.

3. **Error Handling**: Always handle LLM rate limits gracefully with user-friendly messages.

4. **Cache Management**: Clear namespace-specific caches when conversations end to free memory.

5. **Stage Specs**: Use `resolve_stage_specs` for consistent stage configuration.

### For Users

1. **Query Rewrite**: Enable for conversational interfaces with follow-up questions.

2. **Reranking**: Useful when retrieval quality varies; skip for faster responses.

3. **Tools**: Enable for external API access (web search, stock prices, etc.).

4. **Domain Routing**: Use `active_domain` for multi-tenant setups with separate collections.

5. **Processing Steps**: Disable `show_processing_steps` for faster UI without progress indicators.

## Performance Considerations

### Token Budgeting

- Summarization has strict input limits to control costs
- Reranking limits candidates to avoid excessive tokens
- Tools synthesis redacts artifacts to stay within limits

### Caching

- Summary cache significantly reduces costs for repeated queries
- Chunk manager cache improves history access performance
- Idle eviction prevents memory bloat

### Parallel Execution

- Tool calls can be parallel (if LLM supports it)
- Embedding and retrieval are synchronous
- Reranking is sequential after retrieval

## Configuration Example

```python
# backend/core/config.py
class Settings:
    # Database
    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = "documents"
    embedding_model = "openai"
    embedding_model_key = "openai:text-embedding-3-small"
    
    # Retrieval
    top_k = 8
    score_threshold = 0.0
    exact_match = False
    
    # Rewrite
    enable_query_rewrite = True
    rewrite_model = "openai:gpt-4o-mini"
    rewrite_temperature = 0.2
    rewrite_max_output_tokens = 128
    rewrite_tail_turns = 1
    rewrite_summary_turns = 3
    rewrite_confidence_threshold = 0.6
    
    # Summarization
    summarizer_model = "openai:gpt-4o-mini"
    summarizer_temperature = 0.3
    summarizer_max_input_tokens = 512
    summarizer_max_output_tokens = 128
    summary_cache_idle_ttl_seconds = 3600
    
    # Rerank
    re_ranker_model = "openai:gpt-4o-mini"
    re_ranker_temperature = 0.0
    re_ranker_max_output_tokens = 64
    re_ranker_input_rows = 5
    rerank_clear_winner_min_top1 = 0.65
    rerank_clear_winner_min_delta = 0.15
    rerank_exact_match_min_score = 0.80
    reranker_chunk_size = 600
    
    # Inference
    inference_provider = "openai"
    inference_model = "openai:gpt-4o"
    inference_model_key = "openai:gpt-4o"
    inference_temperature = 0.2
    inference_top_p = 1.0
    max_inference_output_tokens = 800
    inference_reasoning_effort = "low"
    debug_thoughts = True
    
    # Tools
    enable_tools = False
    tools_synth_max_output_tokens = 800
    tool_registry_path = "prompts/tool_registry.yaml"
    
    # Web Search
    use_web_search = False
    
    # Display
    display_sources_for_chat = True
    display_sources_for_embed = False
    show_processing_steps = True
    
    # Prompts
    inference_prompt_registry_path = "prompts/prompt_registry.yaml"
    prompt_domain_default = ""
    
    # Domain
    active_domain = ""
    DOMAIN_EMBEDDING_CONFIG = {
        "default": {
            "collection_name": "documents",
            "embedding_model_key": "openai:text-embedding-3-small"
        }
    }
    
    # History
    raw_tail_turns = 10
    chunk_manager_idle_ttl_seconds = 3600
    
    # Debug
    debug_verbose = False
    debug_log_truncate_chars = 4000
    
    # Cost
    cost_basis_tokens = 1_000_000
```

## API Reference

### handle_chat

**Endpoint**: `/chat/{session_id}` (via main.py)

**Method**: POST

**Request Body**:
```json
{
    "message": "What is the capital of France?",
    "history": [],
    "params": {
        "top_k": 8,
        "score_threshold": 0.0,
        "enable_query_rewrite": true,
        "use_tools": false,
        "conversation_id": "conv-123",
        "user_id": "user-456"
    }
}
```

**Response**:
```json
{
    "answer": "The capital of France is Paris.",
    "sources": [
        {
            "payload": {
                "text": "Paris is the capital of France...",
                "section": "Geography",
                "subsection": "France"
            },
            "score": 0.95
        }
    ],
    "metrics": {
        "vectors_retrieved": 8
    },
    "turn_metrics": {
        "embedding": {
            "model": "text-embedding-3-small",
            "input_tokens": 10,
            "cost": 0.00001
        },
        "inference": {
            "model": "gpt-4o",
            "input_tokens": 500,
            "cached_tokens": 0,
            "output_tokens": 100,
            "cost_total": 0.0015
        },
        "totals": {
            "tokens": {"turn_total": 610},
            "cost": {"turn_total": 0.00151}
        }
    },
    "conversation_totals": {
        "tokens": {
            "embedding": 100,
            "llm_input": 2000,
            "llm_output": 500,
            "conversation_total": 2600
        },
        "costs": {
            "embedding": 0.0001,
            "llm_input": 0.006,
            "llm_output": 0.0075,
            "total": 0.0136
        }
    },
    "tools_used": [],
    "rewrite_display": {
        "enabled": true,
        "triggered": false,
        "accepted": false,
        "original": "What is the capital of France?"
    }
}
```

## Troubleshooting

### Common Issues

**Issue**: "Price history response did not include parseable points"
- **Cause**: API response format doesn't match parser expectations
- **Fix**: Update `_extract_points` to handle the actual response structure

**Issue**: High costs per conversation
- **Cause**: Summarization or rerank running too frequently
- **Fix**: Adjust `rewrite_summary_turns`, `rerank_clear_winner_min_top1`, or disable rerank

**Issue**: Slow responses
- **Cause**: Too many stages enabled or high `top_k`
- **Fix**: Disable unnecessary stages, reduce `top_k`, or skip rerank

**Issue**: Rate limit errors
- **Cause**: LLM provider quota exceeded
- **Fix**: Increase quota or switch to a different model/provider

**Issue**: Context not being used
- **Cause**: Low `score_threshold` or poor embeddings
- **Fix**: Lower threshold or improve embedding quality

## Related Files

- `backend/chat/chat_manager.py`: Main chat logic
- `backend/chat/web_search.py`: Web search integration
- `backend/db/qdrant_db.py`: Vector database client
- `backend/embeddings/embeddings_manager.py`: Embedding generation
- `backend/llm/llm_client.py`: LLM API client
- `backend/tools/`: Tool implementations
- `prompts/prompt_registry.yaml`: Prompt templates
- `prompts/tool_registry.yaml`: Tool definitions
- `backend/core/config.py`: Configuration settings
