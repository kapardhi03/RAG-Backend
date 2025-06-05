import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import re
from dataclasses import dataclass

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

from app.services.llamaindex.engine import LlamaIndexRAGEngine
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Context information for query processing"""
    query: str
    kb_id: str
    intent: str
    entities: List[str]
    keywords: List[str]
    filters: Dict[str, Any]
    expansion_terms: List[str]

@dataclass
class QueryResult:
    """Structured query result"""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    confidence_score: float
    processing_time: float
    query_context: QueryContext
    metadata: Dict[str, Any]

class AdvancedQueryEngine:
    """Advanced query engine with hybrid search, intent analysis, and query optimization"""
    
    def __init__(self, rag_engine: LlamaIndexRAGEngine):
        self.rag_engine = rag_engine
        
        # Query analysis patterns
        self.intent_patterns = {
            'factual': [r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\btell me about\b'],
            'list': [r'\blist\b', r'\bshow me\b', r'\bwhat are\b', r'\benumerate\b'],
            'comparison': [r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b'],
            'procedural': [r'\bhow to\b', r'\bsteps\b', r'\bprocess\b', r'\bmethod\b'],
            'analytical': [r'\banalyze\b', r'\bevaluate\b', r'\bassess\b', r'\bexamine\b'],
            'summary': [r'\bsummarize\b', r'\boverview\b', r'\bmain points\b', r'\bkey\b'],
            'meta': [r'\bwhat documents\b', r'\bwhat files\b', r'\bwhat content\b', r'\bwhat do you have\b']
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            'number': r'\b\d+(?:\.\d+)?\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'file': r'\b[\w.-]+\.[a-zA-Z0-9]+\b'
        }
        
        # Initialize query expansion synonyms
        self.synonym_map = self._load_synonyms()
        
        logger.info("Advanced query engine initialized")
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym mappings for query expansion"""
        # This could be loaded from a file or database in production
        return {
            'document': ['file', 'paper', 'text', 'article', 'report'],
            'data': ['information', 'facts', 'details', 'statistics'],
            'analyze': ['examine', 'study', 'investigate', 'review'],
            'method': ['approach', 'technique', 'procedure', 'process'],
            'result': ['outcome', 'finding', 'conclusion', 'output'],
            'feature': ['characteristic', 'attribute', 'property', 'aspect'],
            'issue': ['problem', 'challenge', 'concern', 'difficulty'],
            'solution': ['answer', 'resolution', 'fix', 'remedy'],
            'benefit': ['advantage', 'gain', 'profit', 'value'],
            'requirement': ['need', 'prerequisite', 'condition', 'specification']
        }
    
    async def analyze_query_intent(self, query: str) -> str:
        """Analyze query to determine user intent"""
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general'
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 
            'where', 'why', 'who', 'which', 'this', 'that', 'these', 'those'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    async def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """Expand query with synonyms and related terms"""
        expansion_terms = []
        
        # Add synonyms for keywords
        for keyword in keywords:
            if keyword in self.synonym_map:
                expansion_terms.extend(self.synonym_map[keyword][:2])  # Limit to 2 synonyms
        
        # Use LLM for advanced expansion if available
        if settings.ENABLE_QUERY_EXPANSION and self.rag_engine.llm:
            try:
                llm_expansion = await self._llm_query_expansion(query)
                expansion_terms.extend(llm_expansion)
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}")
        
        # Remove duplicates and limit expansion
        unique_terms = list(set(expansion_terms))[:settings.MAX_EXPANDED_TERMS]
        
        if unique_terms:
            logger.info(f"Query expanded with terms: {unique_terms}")
        
        return unique_terms
    
    async def _llm_query_expansion(self, query: str) -> List[str]:
        """Use LLM to generate query expansion terms"""
        expansion_prompt = f"""
        Given this user query: "{query}"
        
        Generate 3-5 related terms or synonyms that would help find relevant information.
        Return only the terms, separated by commas, without explanation.
        
        Related terms:
        """
        
        response = await asyncio.to_thread(
            self.rag_engine.llm.complete, expansion_prompt
        )
        
        terms = [term.strip() for term in str(response).split(',')]
        return [term for term in terms if term and len(term) > 2]
    
    def build_filters(
        self, 
        entities: Dict[str, List[str]], 
        intent: str,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build metadata filters based on query analysis"""
        filters = {}
        
        # Add entity-based filters
        if 'file' in entities:
            file_extensions = []
            for file_name in entities['file']:
                if '.' in file_name:
                    ext = file_name.split('.')[-1].lower()
                    file_extensions.append(ext)
            
            if file_extensions:
                filters['file_extension'] = {'$in': file_extensions}
        
        # Add intent-based filters
        if intent == 'meta':
            # For meta queries, we might want to prioritize certain document types
            filters['chunk_index'] = {'$eq': 0}  # First chunks often contain overview info
        
        # Add user-provided filters
        if additional_filters:
            filters.update(additional_filters)
        
        return filters
    
    async def create_query_context(
        self,
        query: str,
        kb_id: str,
        user_filters: Optional[Dict[str, Any]] = None
    ) -> QueryContext:
        """Create comprehensive query context"""
        
        # Analyze intent
        intent = await self.analyze_query_intent(query)
        
        # Extract entities and keywords
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        
        # Expand query
        expansion_terms = await self.expand_query(query, keywords)
        
        # Build filters
        filters = self.build_filters(entities, intent, user_filters)
        
        context = QueryContext(
            query=query,
            kb_id=kb_id,
            intent=intent,
            entities=list(entities.values()) if entities else [],
            keywords=keywords,
            filters=filters,
            expansion_terms=expansion_terms
        )
        
        logger.info(f"Query context created: intent={intent}, keywords={keywords[:3]}")
        return context
    
    async def hybrid_retrieve(
        self,
        context: QueryContext,
        top_k: int = 20
    ) -> List[NodeWithScore]:
        """Perform hybrid retrieval combining vector and keyword search"""
        
        # Get the query engine for this KB
        if context.kb_id not in self.rag_engine.query_engines:
            raise ValueError(f"Knowledge base {context.kb_id} not found")
        
        query_engine = self.rag_engine.query_engines[context.kb_id]
        retriever = query_engine.retriever
        
        # Create enhanced query with expansion terms
        enhanced_query = context.query
        if context.expansion_terms:
            enhanced_query += " " + " ".join(context.expansion_terms)
        
        query_bundle = QueryBundle(query_str=enhanced_query)
        
        # Retrieve nodes
        nodes = await asyncio.to_thread(retriever.retrieve, query_bundle)
        
        # Apply custom scoring based on intent and keywords
        scored_nodes = self._apply_custom_scoring(nodes, context)
        
        # Sort by enhanced score
        scored_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        return scored_nodes[:top_k]
    
    def _apply_custom_scoring(
        self,
        nodes: List[NodeWithScore],
        context: QueryContext
    ) -> List[NodeWithScore]:
        """Apply custom scoring based on query context"""
        
        for node in nodes:
            if node.score is None:
                node.score = 0.0
            
            # Boost score for keyword matches
            text_lower = node.node.text.lower()
            keyword_boost = 0.0
            
            for keyword in context.keywords:
                if keyword in text_lower:
                    keyword_boost += 0.1
                    # Extra boost for exact phrase matches
                    if keyword in context.query.lower() and keyword in text_lower:
                        keyword_boost += 0.05
            
            # Intent-based scoring adjustments
            if context.intent == 'list' and any(word in text_lower for word in ['list', 'items', 'examples']):
                keyword_boost += 0.1
            elif context.intent == 'summary' and any(word in text_lower for word in ['overview', 'summary', 'main']):
                keyword_boost += 0.1
            elif context.intent == 'procedural' and any(word in text_lower for word in ['step', 'process', 'method']):
                keyword_boost += 0.1
            
            # Entity matching boost
            for entity_list in context.entities:
                for entity in entity_list:
                    if entity.lower() in text_lower:
                        keyword_boost += 0.05
            
            # Apply boost (cap at reasonable level)
            node.score += min(keyword_boost, 0.3)
        
        return nodes
    
    async def rerank_results(
        self,
        nodes: List[NodeWithScore],
        query: str,
        final_k: int = 10
    ) -> List[NodeWithScore]:
        """Apply advanced reranking to results"""
        
        if not settings.ENABLE_RERANKING or not settings.COHERE_API_KEY:
            return nodes[:final_k]
        
        try:
            # Create reranker if not already done
            reranker = CohereRerank(
                api_key=settings.COHERE_API_KEY,
                top_n=final_k,
                model="rerank-english-v2.0"
            )
            
            # Apply reranking
            reranked_nodes = await asyncio.to_thread(
                reranker.postprocess_nodes,
                nodes,
                QueryBundle(query_str=query)
            )
            
            logger.info(f"Reranked {len(nodes)} to {len(reranked_nodes)} results")
            return reranked_nodes
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original results: {e}")
            return nodes[:final_k]
    
    def calculate_confidence_score(
        self,
        nodes: List[NodeWithScore],
        context: QueryContext
    ) -> float:
        """Calculate overall confidence score for the answer"""
        
        if not nodes:
            return 0.0
        
        # Average similarity score
        similarity_scores = [node.score or 0.0 for node in nodes]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Keyword coverage score
        query_keywords = set(context.keywords)
        covered_keywords = set()
        
        for node in nodes[:3]:  # Check top 3 nodes
            text_lower = node.node.text.lower()
            for keyword in query_keywords:
                if keyword in text_lower:
                    covered_keywords.add(keyword)
        
        keyword_coverage = len(covered_keywords) / len(query_keywords) if query_keywords else 0.0
        
        # Intent alignment score
        intent_score = 0.5  # Base score
        if context.intent == 'meta' and any('document' in node.node.text.lower() for node in nodes[:3]):
            intent_score = 0.8
        elif context.intent in ['list', 'summary'] and len(nodes) >= 3:
            intent_score = 0.7
        
        # Combine scores
        confidence = (avg_similarity * 0.4 + keyword_coverage * 0.4 + intent_score * 0.2)
        return min(confidence, 1.0)
    
    async def generate_contextual_answer(
        self,
        nodes: List[NodeWithScore],
        context: QueryContext
    ) -> str:
        """Generate answer with context awareness"""
        
        if not nodes:
            return "I don't have any relevant information to answer your question."
        
        # Build context string
        context_parts = []
        for i, node in enumerate(nodes[:5]):  # Use top 5 nodes
            source_info = node.node.metadata.get('filename', f'Document {i+1}')
            context_parts.append(f"[Source {i+1}: {source_info}]\n{node.node.text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create intent-specific prompt
        prompt = self._build_answer_prompt(context, context_text)
        
        # Generate answer using LLM
        try:
            response = await asyncio.to_thread(
                self.rag_engine.llm.complete, prompt
            )
            
            answer = str(response).strip()
            
            # Post-process answer based on intent
            answer = self._post_process_answer(answer, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the answer. Please try rephrasing your question."
    
    def _build_answer_prompt(self, context: QueryContext, context_text: str) -> str:
        """Build intent-specific prompt for answer generation"""
        
        base_prompt = f"""Based on the following context, answer the user's question accurately and comprehensively.

Context:
{context_text}

User Question: {context.query}
Query Intent: {context.intent}
"""
        
        # Add intent-specific instructions
        if context.intent == 'list':
            base_prompt += "\nProvide a well-organized list with clear items. Use bullet points or numbered format."
        elif context.intent == 'summary':
            base_prompt += "\nProvide a comprehensive summary covering the main points. Include key details and insights."
        elif context.intent == 'comparison':
            base_prompt += "\nStructure your answer to clearly highlight similarities and differences. Use comparative language."
        elif context.intent == 'procedural':
            base_prompt += "\nProvide step-by-step instructions or explain the process clearly. Use numbered steps if appropriate."
        elif context.intent == 'analytical':
            base_prompt += "\nProvide a thorough analysis with reasoning, evidence, and conclusions. Be analytical and critical."
        elif context.intent == 'meta':
            base_prompt += "\nProvide information about the available documents and content. Include document types, topics, and scope."
        else:
            base_prompt += "\nProvide a clear, factual answer based on the available information."
        
        base_prompt += "\n\nIf the context doesn't contain sufficient information to answer the question, clearly state this limitation."
        base_prompt += "\nAnswer:"
        
        return base_prompt
    
    def _post_process_answer(self, answer: str, context: QueryContext) -> str:
        """Post-process answer based on intent and quality checks"""
        
        # Check for null/empty responses
        if not answer or answer.lower().strip() in ['null', 'none', '']:
            return "I don't have specific information about that topic in the available documents."
        
        # Intent-specific post-processing
        if context.intent == 'list' and not any(marker in answer for marker in ['•', '-', '1.', '2.', '*']):
            # Try to format as a list if it's not already
            if '\n' in answer:
                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                if len(lines) > 2:
                    answer = '\n'.join([f"• {line}" for line in lines])
        
        # Quality checks
        if len(answer) < 50 and context.intent in ['summary', 'analytical']:
            answer += "\n\nNote: The available information is limited. For a more comprehensive answer, additional context may be needed."
        
        return answer
    
    async def execute_advanced_query(
        self,
        query: str,
        kb_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: Optional[bool] = None
    ) -> QueryResult:
        """Execute complete advanced query pipeline"""
        
        start_time = time.time()
        
        try:
            # Step 1: Create query context
            context = await self.create_query_context(query, kb_id, filters)
            
            # Step 2: Hybrid retrieval
            nodes = await self.hybrid_retrieve(context, top_k * 2)  # Get more for reranking
            
            # Step 3: Reranking (if enabled)
            should_rerank = enable_reranking if enable_reranking is not None else settings.ENABLE_RERANKING
            if should_rerank:
                nodes = await self.rerank_results(nodes, query, top_k)
            else:
                nodes = nodes[:top_k]
            
            # Step 4: Calculate confidence
            confidence = self.calculate_confidence_score(nodes, context)
            
            # Step 5: Generate answer
            answer = await self.generate_contextual_answer(nodes, context)
            
            # Step 6: Format sources
            sources = self._format_sources(nodes)
            
            # Step 7: Build result
            processing_time = time.time() - start_time
            
            result = QueryResult(
                answer=answer,
                sources=sources,
                context_used=f"Used {len(nodes)} sources from {kb_id}",
                confidence_score=confidence,
                processing_time=processing_time,
                query_context=context,
                metadata={
                    "total_nodes_retrieved": len(nodes),
                    "reranking_enabled": should_rerank,
                    "expansion_terms_used": len(context.expansion_terms),
                    "filters_applied": bool(context.filters)
                }
            )
            
            logger.info(f"Advanced query completed in {processing_time:.2f}s, confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Advanced query failed: {e}")
            
            # Return error result
            return QueryResult(
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                context_used="Error occurred",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                query_context=QueryContext(query, kb_id, "error", [], [], {}, []),
                metadata={"error": str(e)}
            )
    
    def _format_sources(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """Format source nodes for response"""
        sources = []
        
        for i, node in enumerate(nodes):
            source = {
                "index": i + 1,
                "content": node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
                "metadata": node.node.metadata,
                "score": node.score or 0.0,
                "node_id": node.node.node_id
            }
            
            # Add source identification
            filename = node.node.metadata.get('filename', 'Unknown')
            chunk_index = node.node.metadata.get('chunk_index', 0)
            source['source_ref'] = f"{filename} (Section {chunk_index + 1})"
            
            sources.append(source)
        
        return sources
    
    async def explain_query_processing(self, query: str, kb_id: str) -> Dict[str, Any]:
        """Provide detailed explanation of how a query would be processed"""
        
        context = await self.create_query_context(query, kb_id)
        
        explanation = {
            "original_query": query,
            "detected_intent": context.intent,
            "extracted_keywords": context.keywords,
            "extracted_entities": context.entities,
            "expansion_terms": context.expansion_terms,
            "filters_to_apply": context.filters,
            "processing_strategy": self._get_processing_strategy(context),
            "expected_source_types": self._predict_source_types(context)
        }
        
        return explanation
    
    def _get_processing_strategy(self, context: QueryContext) -> str:
        """Describe the processing strategy for this query"""
        
        strategies = []
        
        if context.expansion_terms:
            strategies.append("Query expansion enabled")
        
        if context.filters:
            strategies.append("Metadata filtering applied")
        
        if context.intent == 'meta':
            strategies.append("Meta-information retrieval")
        elif context.intent in ['list', 'summary']:
            strategies.append("Multi-document synthesis")
        else:
            strategies.append("Standard semantic search")
        
        if settings.ENABLE_RERANKING:
            strategies.append("Advanced reranking")
        
        return " + ".join(strategies)
    
    def _predict_source_types(self, context: QueryContext) -> List[str]:
        """Predict what types of sources would be most relevant"""
        
        source_types = []
        
        if 'file' in [entity for entity_list in context.entities for entity in entity_list]:
            source_types.append("Specific documents")
        
        if context.intent == 'procedural':
            source_types.append("Instructional content")
        elif context.intent == 'analytical':
            source_types.append("Detailed analysis documents")
        elif context.intent == 'summary':
            source_types.append("Overview and summary sections")
        elif context.intent == 'list':
            source_types.append("Structured information")
        else:
            source_types.append("General content")
        
        return source_types or ["All available sources"]
    
    async def batch_query(
        self,
        queries: List[str],
        kb_id: str,
        max_concurrent: int = 3
    ) -> List[QueryResult]:
        """Process multiple queries concurrently"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_single_query(query: str):
            async with semaphore:
                result = await self.execute_advanced_query(query, kb_id)
                results.append(result)
                return result
        
        # Create tasks
        tasks = [process_single_query(query) for query in queries]
        
        # Execute with progress
        for i, task in enumerate(asyncio.as_completed(tasks)):
            await task
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        logger.info(f"Batch query completed: {len(results)} queries processed")
        return results
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get query engine statistics and configuration"""
        return {
            "intent_patterns": list(self.intent_patterns.keys()),
            "entity_types": list(self.entity_patterns.keys()),
            "synonym_categories": len(self.synonym_map),
            "hybrid_search_enabled": settings.ENABLE_HYBRID_SEARCH,
            "query_expansion_enabled": settings.ENABLE_QUERY_EXPANSION,
            "reranking_enabled": settings.ENABLE_RERANKING,
            "max_expansion_terms": settings.MAX_EXPANDED_TERMS,
            "available_knowledge_bases": list(self.rag_engine.indices.keys())
        }