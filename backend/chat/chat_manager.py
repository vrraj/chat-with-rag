from typing import List, Dict, Any
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
    print(f"[DEBUG] Successfully connected to OpenAI. Available models: {len(models.data) if hasattr(models, 'data') else 0}")
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
        """Summarize chat history to keep context manageable"""
        try:
            prompt = (
                "Summarize the following conversation in a few sentences:\n\n"
                + "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            )
            resp = client.responses.create(
                model=settings.inference_model,
                input=prompt,
                max_output_tokens=100,
                temperature=0.7,
            )
            return _extract_text_from_responses(resp)
        except Exception as e:
            print(f"Error summarizing history: {e}")
            return "".join([msg["content"] for msg in history[-3:]])  # Fallback to last 3 messages

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

    def chat(self, message: str, context: List[Dict], use_web_search: bool = False) -> Dict:
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
        
        try:
            print("\n[DEBUG] Getting context from embeddings...")
            search_context = self._get_context(message)
            print(f"[DEBUG] Found {len(search_context)} context items from embeddings")
            
            web_context = []
            if use_web_search:
                print("[DEBUG] Getting web search context...")
                web_context = self._get_web_context(message, search_context)
                print(f"[DEBUG] Found {len(web_context)} web search results")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can answer questions based on provided context.\n" +
                "If you have web search results, use them to provide more accurate answers.\n" +
                "Always cite your sources in your response."}
            ]
            
            if self.chat_history:
                print("[DEBUG] Summarizing chat history...")
                summary = self._summarize_history(self.chat_history)
                messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
            
            print("[DEBUG] Formatting context...")
            context_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(search_context)])
            messages.append({"role": "system", "content": f"Context:\n{context_text}"})
            
            if web_context:
                print("[DEBUG] Formatting web context...")
                web_text = "\n".join([f"{i+1}. {item['title']}\n{item['snippet']}\nURL: {item['url']}" 
                                    for i, item in enumerate(web_context, start=1)])
                messages.append({"role": "system", "content": f"Web search results:\n{web_text}"})
            
            messages.append({"role": "user", "content": message})
            
            print("\n[DEBUG] Sending request to OpenAI...")
            print(f"[DEBUG] Model: {settings.inference_model}")
            print(f"[DEBUG] Messages: {json.dumps(messages, indent=2)}")
            
            try:
                response = client.chat.completions.create(
                    model=settings.inference_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                )
                print(f"\n[DEBUG] Successfully received response from OpenAI")
                if hasattr(response, 'choices') and response.choices:
                    print(f"[DEBUG] Response contains {len(response.choices)} choices")
                    if hasattr(response.choices[0], 'message') and response.choices[0].message:
                        print(f"[DEBUG] First choice has message with content: {response.choices[0].message.content[:200]}...")
                else:
                    print("[DEBUG] Response has no choices or choices are empty")
                    print(f"[DEBUG] Response object: {response}")
                    
            except Exception as e:
                print(f"[ERROR] Exception in OpenAI API call: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
            
            self.chat_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.choices[0].message.content}
            ])
            
            result = {
                "response": response.choices[0].message.content,
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
    params: Dict[str, Any] = (payload or {}).get("params") or {}
    top_k: int = params.get("top_k") or settings.top_k
    score_threshold: float = (
        params.get("score_threshold") if params.get("score_threshold") is not None else settings.score_threshold
    )

    if not message:
        return {"answer": "", "metrics": {"vectors_retrieved": 0}}

    db = QdrantDB(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.collection_name,
    )

    results = db.search_similar(
        query=message,
        limit=int(top_k) if top_k is not None else settings.top_k,
        score_threshold=float(score_threshold) if score_threshold is not None else settings.score_threshold,
        with_vectors=False,
        with_payload=True,
        exact=True,
    )
    print(f"[QDRANT] Search returned {len(results)} results")
    for result in results:
        print(f"[QDRANT] Result {result['id']} - Score: {result['score']:.4f}")
    n = len(results) if results is not None else 0
    print(f"[QDRANT] Number of results: {n}")

    # Optional rerank using configured model; keep top few (small prompt)
    kept = min(3, n)
    metrics: Dict[str, Any] = {"vectors_retrieved": n}
    print(f"[QDRANT] Kept {kept} results")
    # Initialize rerank metrics
    metrics["rerank_tokens"] = 0
    metrics["rerank_cost"] = 0.0
    # Rerank if more than one result
    reranked = results
    if n > 1:
        print(f"[QDRANT] Reranking {n} results")
        try:
            # Prepare compact candidate list for ranking
            cand_text = [
                (res.get("payload") or {}).get("text")
                or (res.get("payload") or {}).get("snippet")
                or (res.get("payload") or {}).get("content")
                or ""
                for res in results
            ]
            prompt_text = (
                "Rerank candidates by relevance to the query.\n"
                "Return ONLY a JSON array of indices (0-based) in descending relevance.\n\n"
                f"Query: {message}\n\nCandidates:\n"
                + "\n".join([f"[{i}] {t[:500]}" for i, t in enumerate(cand_text)])
            )
            print(f"[QDRANT] Rerank prompt: {prompt_text}")
            try:
                print(f"[DEBUG] Sending rerank request (Responses API) with model: {settings.re_ranker_model}")
                resp_rerank = client.responses.create(
                    model=settings.re_ranker_model,
                    input=prompt_text,
                    max_output_tokens=64,
                    temperature=0,
                )
                content = _extract_text_from_responses(resp_rerank).strip()
                print(f"[DEBUG] Rerank raw text: {content[:200]}")

                try:
                    # Clean the content by removing markdown code block markers if present
                    if content.startswith('```json') and content.endswith('```'):
                        content = content[7:-3].strip()
                    elif content.startswith('```') and content.endswith('```'):
                        content = content[3:-3].strip()

                    order = json.loads(content)
                    print(f"[DEBUG] Successfully parsed JSON response")
                except json.JSONDecodeError as je:
                    print(f"[WARNING] Failed to parse JSON response: {je}")
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1 and start < end:
                        try:
                            order = json.loads(content[start:end+1])
                            print(f"[DEBUG] Successfully extracted JSON array from response")
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
            print(f"[QDRANT] Reranked results: {reranked}")

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
    # Build compact prompt from top reranked chunks and user message
    print(f"[QDRANT] Call to Inference API with Reranked results: {reranked}")
    try:
        ctx_bits: List[str] = []
        for item in (reranked or [])[:kept]:
            payload = item.get("payload") or {}
            title = payload.get("title") or payload.get("metadata", {}).get("title") or ""
            text = payload.get("text") or payload.get("snippet") or payload.get("content") or ""
            snippet = text.strip().replace("\n", " ")[:200]
            bit = f"{title} {snippet}".strip()
            ctx_bits.append(bit)
        # Store kept texts as context list for the next step
        context = ctx_bits[:]
        context_inline = " | ".join([b for b in ctx_bits if b])
        print(f"[QDRANT] Context: {context_inline}")

        compact_prompt = (
            f"Answer concisely using context if relevant. "
            f"Context: {context_inline} "
            f"Question: {message}"
        ).strip()
        print(f"[DEBUG] Compact prompt: {compact_prompt}")
        
        try:
            print(f"[DEBUG] Attempting Responses API with Inference model: {settings.inference_model}")
            resp_inf = client.responses.create(
                model=settings.inference_model,
                input=compact_prompt,
                max_output_tokens=int(getattr(settings, "max_output_tokens", 300)),
                temperature=0.7,
            )
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
        print(f"[INFERENCE] Extracted answer: {answer}")

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

    return {"answer": answer, "metrics": metrics}
