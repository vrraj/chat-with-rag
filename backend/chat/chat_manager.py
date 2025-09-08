from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)
import json
from openai import OpenAI
from backend.core.config import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.db.qdrant_db import QdrantDB
from backend.chat.web_search import WebSearchClient

# Single OpenAI client for this module
print(f"[DEBUG] Initializing OpenAI client with API key: {'*' * 8 + settings.openai_api_key[-4:] if settings.openai_api_key else 'None'}")
client = OpenAI(api_key=settings.openai_api_key)

# Test the client can be initialized
try:
    print("[DEBUG] Testing OpenAI client...")
    models = client.models.list()
    #print(f"[DEBUG] Successfully connected to OpenAI. Available models: {len(models.data) if hasattr(models, 'data') else 0}")
except Exception as e:
    print(f"[ERROR] Failed to initialize OpenAI client: {str(e)}")
    import traceback
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    raise


def _extract_text_from_responses(resp) -> str:
    """Return response text from Responses API object.

    Prefers `resp.output_text`. If absent, concatenates any `.text` parts
    from `resp.output[...].content[...]` entries. Falls back to empty string.
    """
    logger.debug(f"Full response object: {resp}")
    # Prefer direct output_text if available
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # Try to read from output -> content -> text
    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")

    parts: List[str] = []
    if output and isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not content or not isinstance(content, list):
                continue
            for c in content:
                txt = getattr(c, "text", None)
                if txt is None and isinstance(c, dict):
                    txt = c.get("text")
                if isinstance(txt, str) and txt:
                    parts.append(txt)

    return "".join(parts) if parts else ""



def _extract_usage_from_responses(resp) -> Dict[str, int] | None:
    """Extract usage fields (prompt, completion, total tokens) if present.

    Returns a dict with keys: prompt_tokens, completion_tokens, total_tokens;
    otherwise None if usage is unavailable.
    """
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return None

    def _get(u, name):
        return getattr(u, name, None) if not isinstance(u, dict) else u.get(name)

    p = _get(usage, "prompt_tokens")
    c = _get(usage, "completion_tokens")
    t = _get(usage, "total_tokens")

    if p is None and c is None and t is None:
        return None

    result: Dict[str, int] = {}
    if p is not None:
        result["prompt_tokens"] = int(p)
    if c is not None:
        result["completion_tokens"] = int(c)
    if t is not None:
        result["total_tokens"] = int(t)
    return result

# --- Simple utilities to collapse duplicate sources by URL + payload fields ---

def _render_source_line(indices: list[int], url: str, section: str, subsection: str) -> str:
    idx_text = ", ".join(str(i) for i in sorted(set(indices)))
    return f"[{idx_text}] {url} (Section: {section} > {subsection})"


def _collapse_sources(indexed_items: List[Dict[str, Any]]) -> str:
    """Group by (url, section, subsection) and collapse indices.
    `indexed_items` items look like: {index:int, url:str, section:str, subsection:str}
    Returns a single string with one line per unique group.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for it in indexed_items:
        url = (it.get("url") or "unknown").strip()
        section = (it.get("section") or "N/A").strip()
        subsection = (it.get("subsection") or "N/A").strip()
        key = (url, section, subsection)
        if key not in groups:
            groups[key] = {"indices": [], "url": url, "section": section, "subsection": subsection}
        idx = int(it.get("index", 0) or 0)
        if idx > 0:
            groups[key]["indices"].append(idx)

    lines: List[str] = []
    for (_url, _section, _subsection), data in groups.items():
        if data["indices"]:
            lines.append(_render_source_line(data["indices"], data["url"], data["section"], data["subsection"]))
    return "\n".join(lines)
# --- end utilities ---

class ChatManager:
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name
        )
        self.web_search = WebSearchClient()
        self.chat_history = []
        self.max_tokens = settings.max_history_tokens

    

    def _summarize_history(self, history: List[Dict]) -> str:
        """Summarize recent chat history to keep context manageable.

        Uses the last `settings.chat_history_window_turns * 2` messages
        (user + assistant per turn) when building the summary prompt.
        """
        try:
            # Determine how many recent messages to include (2 per turn)
            turns = max(1, int(getattr(settings, "chat_history_window_turns", 3)))
            n_messages = max(2, turns * 2)
            recent_history = history[-n_messages:] if history else []
            prompt = (
                "Summarize the following conversation in a few sentences:\n\n"
                + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            )
            resp = client.responses.create(
                model=settings.summarize_model,
                input=prompt,
                max_output_tokens=getattr(settings, "summarize_max_inference_output_tokens", getattr(settings, "summarize_max_output_tokens", 500)),
                temperature=settings.summarizer_temperature
            )
            return _extract_text_from_responses(resp)
        except Exception as e:
            print(f"Error summarizing history: {e}")
            turns = max(1, int(getattr(settings, "chat_history_window_turns", 3)))
            n_messages = max(2, turns * 2)
            recent = history[-n_messages:] if history else []
            # Fallback to concatenation of recent message contents
            return "".join([msg.get("content", "") for msg in recent])

    def _get_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Get relevant context from QdrantDB"""
        print(f"[DEBUG] Searching Qdrant for query: {query}")
        try:
            results = self.qdrant_db.search_similar(query, limit=limit)
            print(f"[DEBUG] Qdrant search returned {len(results)} results")
            if results:
                print(f"[DEBUG] First result score: {results[0].get('score', 'N/A')}")
            return results
        except Exception as e:
            print(f"[ERROR] Error in _get_context: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return []

    def _get_web_context(self, query: str, existing_context: List[Dict]) -> List[Dict]:
        """Get additional context from web search"""
        return self.web_search.get_additional_context(query, existing_context)

    def chat(self, message: str, context: List[Dict], use_web_search: bool = False, params: Dict[str, Any] | None = None) -> Dict:
        """
        Process a chat message and generate a response
        Args:
            message: User's message
            context: Additional context to consider
            use_web_search: Whether to use web search for additional context
        Returns:
            Chat response with sources
        """
        print(f"\n=== Starting chat with message: {message} ===")
        print(f"Context length: {len(context)}, use_web_search: {use_web_search}")
        print (f"Params: {params}")
        try:
            print("\n[DEBUG] Getting context from embeddings...")
            search_context = self._get_context(message)
            print(f"[DEBUG] Found {len(search_context)} context items from embeddings")
            
            web_context = []
            if use_web_search:
                print("[DEBUG] Getting web search context...")
                web_context = self._get_web_context(message, search_context)
                print(f"[DEBUG] Found {len(web_context)} web search results")
            
            system_prompt = (
                "You are a helpful assistant. Use the provided context to answer the user's question.\n"
                "If any context chunk has a citation like [1], [2], etc., retain it in your response.\n"
                "Do not fabricate sources. If no source supports the answer, say so clearly.\n"
                "If a source URL is available (shown in the final 'Sources' section), consider referencing it by its tag like [1]."
            )
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if self.chat_history:
                print("[DEBUG] Summarizing chat history...")
                summary = self._summarize_history(self.chat_history)
                messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
            
            print("[DEBUG] Formatting context...")
            context_text = "\n".join([
                f"[{i+1}] {c['text']} (Section: {c.get('section', 'N/A')} > {c.get('subsection', 'N/A')})"
                for i, c in enumerate(search_context)
            ])
            messages.append({"role": "system", "content": f"Context:\n{context_text}"})
            
            if web_context:
                print("[DEBUG] Formatting web context...")
                web_text = "\n".join([f"{i+1}. {item['title']}\n{item['snippet']}\nURL: {item['url']}" 
                                    for i, item in enumerate(web_context, start=1)])
                messages.append({"role": "system", "content": f"Web search results:\n{web_text}"})
            
            messages.append({"role": "user", "content": message})
            
            #print("\n[DEBUG] Sending request to OpenAI...")
            print(f"[DEBUG] Model: {settings.inference_model}")
            #print(f"[DEBUG] Messages: {json.dumps(messages, indent=2)}")
            
            from backend.utils.prompt_utils import convert_messages_to_prompt
            prompt = convert_messages_to_prompt(messages)
            try:
                params = params or {}
                # Pull overrides from params with flexible keys
                def pick(keys, default=None):
                    for k in keys:
                        if k in params and params[k] is not None:
                            return params[k]
                    return default

                temperature = pick(["temperature", "inference_temperature", "INFERENCE_TEMPERATURE"], getattr(settings, "inference_temperature", 0.7))
                max_out = pick(["max_output_tokens", "max_inference_output_tokens", "MAX_INFERENCE_OUTPUT_TOKENS"], getattr(settings, "max_inference_output_tokens", 500))
                top_p = pick(["top_p", "inference_top_p", "INFERENCE_TOP_P"], getattr(settings, "inference_top_p", None))

                _kwargs = {
                    "model": settings.inference_model,
                    "input": prompt,
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_out),
                }
                # Optional decoding controls if present
                if top_p is not None:
                    _kwargs["top_p"] = float(top_p)
                # Reasoning effort: controlled solely via settings
                if getattr(settings, "inference_reasoning_model", False):
                    _kwargs["reasoning"] = {"effort": getattr(settings, "inference_reasoning_effort", "low")}
                # Invoke Responses API 
                response = client.responses.create(**_kwargs)
                print(f"\n[DEBUG] Successfully received response from OpenAI")
                logger.debug(f"[DEBUG] Raw Responses inference: {response}")
            except Exception as e:
                print(f"[ERROR] Exception in OpenAI API call: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
            
            from backend.utils.prompt_utils import convert_messages_to_prompt
            # Extract answer text using the helper function
            answer = _extract_text_from_responses(response) or ""
            self.chat_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer}
            ])

            # Build Sources (collapse duplicates by URL + payload fields)
            indexed = [
                {
                    "index": i + 1,
                    "url": c.get("url_lower", c.get("url", "unknown")),
                    "section": c.get("section", "N/A"),
                    "subsection": c.get("subsection", "N/A"),
                }
                for i, c in enumerate(search_context)
            ]
            collapsed = _collapse_sources(indexed)
            source_notes = "\n\nSources:\n" + collapsed if collapsed else ""

            # Append web sources (kept separate; no dedupe against numeric indices)
            if web_context:
                web_notes = "\n" + "\n".join([
                    f"[web-{i+1}] {item.get('url', 'Web result')}" for i, item in enumerate(web_context)
                ])
                source_notes += web_notes

            result = {
                # Only strip trailing newlines, not meaningful citation text
                "response": answer.rstrip("\n") + source_notes,
                "sources": search_context + web_context
            }

            print(f"\n[DEBUG] Returning response with {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\n[ERROR] Exception in chat: {e}")
            print(f"[ERROR] Traceback: {error_trace}")
            return {
                "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}", 
                "sources": []
            }


def handle_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Thin handler: fetch candidates from Qdrant only.

    Expects payload with keys: message, params.top_k, params.score_threshold.
    Returns a summary and basic metrics without rerank or LLM.
    """
    message: str = (payload or {}).get("message") or ""
    history: List[Dict[str, str]] = (payload or {}).get("history") or []
    params: Dict[str, Any] = (payload or {}).get("params") or {}
    top_k: int = params.get("top_k") or settings.top_k
    score_threshold: float = params.get("score_threshold") or settings.score_threshold
    
    if not message:
        return {"answer": "", "metrics": {"vectors_retrieved": 0}}

    db = QdrantDB(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.collection_name,
    )

    results = db.search_similar(
        query=message,
        limit=int(top_k),
        score_threshold=float(score_threshold),
        with_vectors=False,
        with_payload=True,
        exact=True,
    )
    #print(f"[QDRANT] Search returned {len(results)} results")
    #for result in results:
        #print(f"[QDRANT] Result {result['id']} - Score: {result['score']:.4f}")
    n = len(results) if results is not None else 0
    #print(f"[QDRANT] Number of results: {n}")

    # Optional rerank using configured model; keep top few (small prompt)
    kept = min(settings.re_ranker_input_rows, n) # Keep top N results for reranking to a maximum of settings.re_ranker_input_rows
    metrics: Dict[str, Any] = {"vectors_retrieved": n}
    #print(f"[QDRANT] Kept {kept} results")
    # Initialize rerank metrics
    metrics["rerank_tokens"] = 0
    metrics["rerank_cost"] = 0.0
    # Rerank if more than one result
    reranked = results
    if n > 1:
        print(f"[Chat Manager] Reranking {n} results")
        try:
            # Prepare compact candidate list for ranking - Current payload only has text > Snippet and Content for Future use
            cand_text = [
                (res.get("payload") or {}).get("text")
                or (res.get("payload") or {}).get("snippet")
                or (res.get("payload") or {}).get("content")
                or ""
                for res in results
            ]
            print("[DBG] raw cand repr:", repr(cand_text[0][:300]))
            print("[DBG] raw cand repr:", repr(cand_text[1][:300]))
            prompt_text = (
                "Rerank candidates by relevance to the query.\n"
                "Return ONLY a JSON array of indices (0-based) in descending relevance.\n"
                f"Query: {message}\n\nCandidates:\n"
                + "\n".join([f"[{i}] {t[:settings.reranker_chunk_size]}" for i, t in enumerate(cand_text)])
            )
            print(f"[Chat Manager] Rerank prompt: {prompt_text}")
            try:
                #print(f"[DEBUG] Sending rerank request (Responses API) with model: {settings.re_ranker_model}")
                resp_rerank = client.responses.create(
                    model=settings.re_ranker_model,
                    input=prompt_text.strip(),
                    max_output_tokens=settings.re_ranker_max_output_tokens,
                    temperature=settings.re_ranker_temperature
                )
                content = _extract_text_from_responses(resp_rerank).strip()
                print(f"[Chat Manager] Rerank raw text: {content[:200]}")

                try:
                    # Clean the content by removing markdown code block markers if present
                    if content.startswith('```json') and content.endswith('```'):
                        content = content[7:-3].strip()
                    elif content.startswith('```') and content.endswith('```'):
                        content = content[3:-3].strip()

                    order = json.loads(content)
                    #print(f"[DEBUG] Successfully parsed JSON response")
                except json.JSONDecodeError as je:
                    print(f"[WARNING] Failed to parse JSON response: {je}")
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1 and start < end:
                        try:
                            order = json.loads(content[start:end+1])
                            #print(f"[DEBUG] Successfully extracted JSON array from response")
                        except json.JSONDecodeError:
                            print("[WARNING] Could not parse extracted JSON, using original order")
                            order = list(range(n))
                    else:
                        print("[WARNING] Could not extract JSON array from response, using original order")
                        order = list(range(n))
            except Exception as e:
                print(f"[ERROR] Failed to get rerank response: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                print("[INFO] Falling back to original order")
                order = list(range(n))

            # Build reranked list and keep top few
            order = [i for i in order if isinstance(i, int) and 0 <= i < n]
            reranked = [results[i] for i in order] or results
            reranked = reranked[:kept]
            print(f"[Chat Manager] Reranked results: {reranked}")

            # Metrics from usage if available
            usage = _extract_usage_from_responses(resp_rerank)
            if usage:
                prompt_toks = int(usage.get("prompt_tokens", 0) or 0)
                completion_toks = int(usage.get("completion_tokens", 0) or 0)
                total = int(usage.get("total_tokens", prompt_toks + completion_toks) or (prompt_toks + completion_toks))
                metrics["rerank_tokens"] = total
                cost = (
                    prompt_toks * float(settings.re_ranker_cost_per_MM_tokens_input)
                    + completion_toks * float(settings.re_ranker_cost_per_MM_tokens_output)
                ) / 1_000_000.0
                metrics["rerank_cost"] = round(float(cost), 8)
        except Exception:
            reranked = results[:kept]

    # Call to Inference API
    # By default, do NOT use compact prompt; instead, construct a full prompt with context and citations.
    print(f"[Chat Manager] Call to Inference API with Reranked results: {reranked}")
    try:
        # Optional: summarize recent chat history (UI-provided) to give brief context
        summary_text = ""
        if history:
            try:
                # Take last N turns (up to N*2 messages) to keep prompt small
                turns = max(1, int(getattr(settings, "chat_history_window_turns", 3)))
                n_messages = max(2, turns * 2)
                recent = history[-n_messages:]
                hist_prompt = (
                    "Summarize the following conversation in a few sentences:\n\n"
                    + "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent])
                )
                resp_sum = client.responses.create(
                    model=settings.summarize_model,
                    input=hist_prompt,
                    max_output_tokens=getattr(settings, "summarize_max_inference_output_tokens", getattr(settings, "summarize_max_output_tokens", 500)),
                    temperature=settings.summarizer_temperature,
                )
                summary_text = _extract_text_from_responses(resp_sum).strip()
                print(f"[SUMMARY] History summary: {summary_text}")
            except Exception as e:
                print(f"[WARNING] Summarization failed, proceeding without summary: {e}")

        # Build context as full text chunks with citations
        context_citations: List[str] = []
        for i, item in enumerate((reranked or [])[:kept]):
            payload = item.get("payload") or {}
            text = payload.get("text") or payload.get("snippet") or payload.get("content") or ""
            section = payload.get("section", "N/A")
            subsection = payload.get("subsection", "N/A")
            url_lower = payload.get("url_lower", payload.get("url", "unknown"))
            citation = f"[{i+1}] {text.strip()} (Section: {section} > {subsection})"
            context_citations.append(citation)
        context_full = "\n".join(context_citations)
        # Build Sources (collapse duplicates by URL + payload fields)
        indexed_for_collapse = [
            {
                "index": i + 1,
                "url": ((item.get('payload') or {}).get('url_lower', (item.get('payload') or {}).get('url', 'unknown'))),
                "section": (item.get('payload') or {}).get('section', 'N/A'),
                "subsection": (item.get('payload') or {}).get('subsection', 'N/A'),
            }
            for i, item in enumerate((reranked or [])[:kept])
        ]
        sources_section = "\nSources:\n" + _collapse_sources(indexed_for_collapse)
        # Construct the prompt (optionally prepend brief summary)
        summary_block = (f"Previous conversation summary: {summary_text}\n\n" if summary_text else "")
        prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer the user's question. "
            "If any context chunk has a citation like [1], [2], etc., retain it in your response. "
            "Do not fabricate sources. If no source supports the answer, say so clearly. "
            "If a source URL is available (shown in the final 'Sources' section), consider referencing it by its tag like [1].\n\n"
            + summary_block
            + f"Context:\n{context_full}\n\n"
            + f"Question: {message}\n"
        )
        logger.debug(f"[FULL PROMPT] Sent to model:\n{prompt}")
        print(f"[Chat Manager] Full prompt for model:\n{prompt}")
        try:
            print(f"[Chat Manager] Attempting Responses API with Inference model: {settings.inference_model}")
            # Allow per-request overrides via payload params
            params = params or {}
            def pick(keys, default=None):
                for k in keys:
                    if k in params and params[k] is not None:
                        return params[k]
                return default
            temperature = pick(["temperature", "inference_temperature", "INFERENCE_TEMPERATURE"], getattr(settings, "inference_temperature", 0.7))
            max_out = pick(["max_output_tokens", "max_inference_output_tokens", "MAX_INFERENCE_OUTPUT_TOKENS"], getattr(settings, "max_inference_output_tokens", 300))
            top_p = pick(["top_p", "inference_top_p", "INFERENCE_TOP_P"], getattr(settings, "inference_top_p", None))

            _kwargs2 = {
                "model": settings.inference_model,
                "input": prompt,
                "max_output_tokens": int(max_out),
                "temperature": float(temperature),
            }
            if top_p is not None:
                _kwargs2["top_p"] = float(top_p)
            # Reasoning effort: controlled solely via settings
            if getattr(settings, "inference_reasoning_model", False):
                _kwargs2["reasoning"] = {"effort": getattr(settings, "inference_reasoning_effort", "low")}

            resp_inf = client.responses.create(**_kwargs2)
            logger.debug(f"Full response object: {resp_inf}")
            print(f"[DEBUG] Raw Responses inference: {resp_inf}")
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {str(e)}")
            print(f"[ERROR] API Key present: {'Yes' if hasattr(settings, 'openai_api_key') and settings.openai_api_key else 'No'}")
            print(f"[ERROR] Model: {settings.inference_model}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Re-raise to ensure the error is not silently caught
            raise

        # Extract answer text using the helper function
        answer = _extract_text_from_responses(resp_inf) or ""
        print(f"[Chat Manager - INFERENCE] Extracted answer: {answer}")

        # Extract and log usage metrics using the helper function
        usage = _extract_usage_from_responses(resp_inf)
        if usage:
            p_tok = int(usage.get("prompt_tokens", 0) or 0)
            c_tok = int(usage.get("completion_tokens", 0) or 0)
            t_tok = int(usage.get("total_tokens", p_tok + c_tok) or (p_tok + c_tok))

            metrics.update({
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
                "prompt_cost": round((p_tok * float(settings.inference_cost_per_MM_tokens_input)) / 1_000_000.0, 8),
                "completion_cost": round((c_tok * float(settings.inference_cost_per_MM_tokens_output)) / 1_000_000.0, 8)
            })
            metrics["total_cost"] = round(metrics["prompt_cost"] + metrics["completion_cost"], 8)

            print(f"[METRICS] Token usage - Prompt: {p_tok}, Completion: {c_tok}, Total: {t_tok}")
            print(f"[METRICS] Cost - Prompt: ${metrics['prompt_cost']}, Completion: ${metrics['completion_cost']}, Total: ${metrics['total_cost']}")
        else:
            print("[WARNING] No usage metrics available in response")
    except Exception:
        answer = ""

    # Return the answer with sources appended for proper citation
    return {"answer": answer.rstrip("\n") + sources_section, "metrics": metrics}
