from typing import List, Dict, Any
import openai
from backend.core.config import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.db.qdrant_db import QdrantDB
from backend.chat.web_search import WebSearchClient

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

    async def chat(self, message: str, context: List[Dict], use_web_search: bool = False) -> Dict:
        """
        Process a chat message and generate a response
        Args:
            message: User's message
            context: Additional context to consider
            use_web_search: Whether to use web search for additional context
        Returns:
            Chat response with sources
        """
        try:
            # Get relevant context from QdrantDB
            search_context = await self.qdrant_db.search_similar(message, limit=5)
            
            # Get web context if enabled
            web_context = []
            if use_web_search:
                web_context = await self.web_search.get_additional_context(message, search_context)

            # Combine all contexts
            combined_context = search_context + web_context

            # Prepare messages for chat model
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
                {"role": "user", "content": message}
            ]

            # Add context to messages
            if combined_context:
                context_summary = "\n".join([
                    f"{item.get('title', '')}: {item.get('snippet', '')}"
                    for item in combined_context[:3]  # Limit to top 3 for brevity
                ])
                messages.insert(1, {"role": "system", "content": f"Context:\n{context_summary}"})

            # Generate non-streaming response
            response = await openai.ChatCompletion.acreate(
                model=settings.CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )

            # Get the complete response
            complete_response = response.choices[0].message.content
            
            # Update chat history with complete message
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": complete_response})

            # Return complete response with sources
            return {
                "response": complete_response,
                "sources": combined_context
            }

        except Exception as e:
            print(f"Error in chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _summarize_history(self, history: List[Dict]) -> str:
        """Summarize chat history to keep context manageable"""
        try:
            response = openai.ChatCompletion.create(
                model=settings.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "Summarize the following conversation in a few sentences:"},
                    {"role": "user", "content": "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing history: {e}")
            return "".join([msg["content"] for msg in history[-3:]])  # Fallback to last 3 messages

    def _get_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Get relevant context from QdrantDB"""
        return self.qdrant_db.search_similar(query, limit=limit)

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
        try:
            # Get relevant context from embeddings
            search_context = self._get_context(message)
            
            # Get web search results if enabled
            web_context = []
            if use_web_search:
                web_context = self._get_web_context(message, search_context)
            
            # Prepare messages for GPT
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can answer questions based on provided context.\n" +
                "If you have web search results, use them to provide more accurate answers.\n" +
                "Always cite your sources in your response."}
            ]
            
            # Add chat history
            if self.chat_history:
                summary = self._summarize_history(self.chat_history)
                messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
            
            # Add context
            context_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(search_context)])
            messages.append({"role": "system", "content": f"Context:\n{context_text}"})
            
            # Add web search results if available
            if web_context:
                web_text = "\n".join([f"{i+1}. {item['title']}\n{item['snippet']}\nURL: {item['url']}" 
                                  for i, item in enumerate(web_context, start=1)])
                messages.append({"role": "system", "content": f"Web search results:\n{web_text}"})
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Get response from GPT
            response = openai.ChatCompletion.create(
                model=settings.CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Update chat history
            self.chat_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.choices[0].message.content}
            ])
            
            # Return response with all sources
            return {
                "response": response.choices[0].message.content,
                "sources": search_context + web_context
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return {"response": "I'm sorry, I encountered an error while processing your request.", "sources": []}
