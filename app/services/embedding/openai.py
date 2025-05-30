# app/services/embedding/openai_fixed.py - Compatibility fix for older OpenAI versions
import logging
import os
import openai
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# Setup logging
logger = logging.getLogger("embedding_service")

class OpenAIEmbedding:
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002", chat_model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI embedding service"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Set the API key directly for v0.28 style
        openai.api_key = self.api_key
        
        self.model = model
        self.chat_model = chat_model
        self.logger = logging.getLogger("embedding_service")
        self.logger.info(f"OpenAI service initialized with old API")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using OpenAI API"""
        try:
            self.logger.info(f"Generating embeddings for batch of {len(batch)} texts")
            
            # Call OpenAI API using v0.28 style syntax
            response = await openai.Embedding.acreate(
                model=self.model,
                input=batch
            )
            
            # Extract embeddings from response (v0.28 style)
            embeddings = [data["embedding"] for data in response["data"]]
            
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def embed_documents(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} documents")
            
            # Process documents in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} texts)")
                
                try:
                    batch_embeddings = await self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    self.logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    raise ValueError(f"Embedding generation failed at batch {i//batch_size + 1}: {str(e)}")
            
            self.logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
        
        except Exception as e:
            self.logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query text"""
        try:
            self.logger.info(f"Generating embedding for query")
            
            # Call OpenAI API (v0.28 style)
            response = await openai.Embedding.acreate(
                model=self.model,
                input=[query]
            )
            
            # Extract embedding from response (v0.28 style)
            embedding = response["data"][0]["embedding"]
            
            self.logger.info(f"Successfully generated query embedding")
            return embedding
        
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def _build_simple_system_prompt(self, intent_info: Dict, relevance_info: Dict) -> str:
        """Build a simple system prompt"""
        
        base_prompt = """You are a helpful AI assistant that answers questions based on provided documents. 

Instructions:
- Answer questions using ONLY the information in the provided context
- Be accurate and specific
- If the context doesn't contain relevant information, say so clearly, just 'null'
- Organize your response with clear structure when helpful
- Always be truthful about what information is and isn't available"""

        # Add intent-specific instructions
        if intent_info.get("is_meta_query"):
            base_prompt += """

This is a question about what documents or information you have available. Provide a helpful overview of the content in your knowledge base."""

        elif intent_info.get("is_list_request"):
            base_prompt += """

This is a request for a list. Provide a well-organized list with relevant details."""

        elif intent_info.get("is_summary_request"):
            base_prompt += """

This is a request for a summary. Provide a comprehensive overview of the main points."""

        return base_prompt
    
    def _build_simple_user_prompt(self, query: str, context: str) -> str:
        """Build a simple user prompt"""
        
        if not context or context.strip() == "":
            return f"Question: {query}\n\nContext: No relevant documents found.\n\nPlease respond that no relevant information is available."
        
        return f"""Context from documents:
{context}

Question: {query}

Please answer the question using only the information provided in the context above. If the context doesn't contain relevant information for the question, please reply only 'null'."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def generate_advanced_answer(
        self, 
        query: str, 
        context: str, 
        intent_info: Dict[str, Any],
        relevance_info: Dict[str, Any],
        kb_id: str,
        max_tokens: int = 1500
    ) -> str:
        """Generate an answer using the older OpenAI API"""
        try:
            # Check if we have any context at all
            if not context or context.strip() == "":
                return "I don't have any documents or information in this knowledge base to answer your question."
            
            # Build prompts
            system_prompt = self._build_simple_system_prompt(intent_info, relevance_info)
            user_prompt = self._build_simple_user_prompt(query, context)
            
            self.logger.info(f"Generating answer with confidence: {relevance_info.get('confidence', 'unknown')}")
            
            # Use lower temperature for more factual responses
            temperature = 0.1
            if intent_info.get("is_summary_request"):
                temperature = 0.2
            
            # Generate response using old API
            response = await openai.ChatCompletion.acreate(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            answer = response['choices'][0]['message']['content'].strip()
            
            # Post-process the answer
            answer = self._post_process_answer(answer, intent_info, relevance_info)
            
            self.logger.info(f"Generated answer of {len(answer)} characters")
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            # Fallback response
            if relevance_info.get("has_relevant_content"):
                return "I found relevant information but encountered an error generating the response. Please try rephrasing your question."
            else:
                return "I don't have specific information about that topic in the available documents."

    def _post_process_answer(self, answer: str, intent_info: Dict, relevance_info: Dict) -> str:
        """Post-process the generated answer"""
        
        # Check for "I don't know" patterns
        no_info_patterns = [
            "i don't have", "i cannot find", "i couldn't find", "not mentioned",
            "doesn't contain", "does not contain", "no information",
            "not provided", "not available", "cannot determine"
            "I don't have specific information about that topic in the available documents."
        ]
        
        answer_lower = answer.lower()
        
        if any(pattern in answer_lower for pattern in no_info_patterns):
            return "null"
        
        # If the answer is mostly negative and we don't have relevant content
        if (any(pattern in answer_lower for pattern in no_info_patterns) and 
            not relevance_info.get("has_relevant_content", False) and
            len(answer) < 200):
            return "null"
        
        return answer

    # Keep backward compatibility
    async def generate_enhanced_answer(self, query: str, context: str, **kwargs):
        """Backward compatibility wrapper"""
        intent_info = {"question_type": "general"}
        relevance_info = {"has_relevant_content": bool(context), "confidence": "medium"}
        
        return await self.generate_advanced_answer(
            query=query,
            context=context,
            intent_info=intent_info,
            relevance_info=relevance_info,
            kb_id="unknown"
        )