"""
Reference-free evaluation metrics for RAG systems.
These metrics don't require ground truth labels.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
import re


class RetrievalMetrics(BaseModel):
    """Metrics for evaluating retrieval quality."""
    num_sources: int  # Number of sources retrieved
    avg_source_length: float  # Average length of source documents
    source_diversity: float  # How diverse are the sources (based on unique words)
    

class AnswerMetrics(BaseModel):
    """Metrics for evaluating answer quality."""
    answer_length: int  # Length of the generated answer
    citation_count: int  # Number of times sources are cited/referenced
    answer_source_overlap: float  # Overlap between answer and source content (0-1)
    

class FaithfulnessMetrics(BaseModel):
    """Metrics for evaluating faithfulness to sources."""
    has_citations: bool  # Does the answer reference sources?
    claims_count: int  # Number of factual claims made
    # Note: For real faithfulness, you'd use an LLM-as-judge here
    

class EvaluationResult(BaseModel):
    """Complete evaluation result for a single query."""
    query: str
    workflow_id: str
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    faithfulness: FaithfulnessMetrics
    response: str
    sources: List[Dict[str, Any]]


class MetricsCalculator:
    """Calculate reference-free metrics for RAG evaluation."""
    
    @staticmethod
    def calculate_retrieval_metrics(sources: List[Dict[str, Any]]) -> RetrievalMetrics:
        """
        Calculate metrics about the retrieved sources.
        
        Args:
            sources: List of source documents with 'title', 'url', etc.
            
        Returns:
            RetrievalMetrics
        """
        if not sources:
            return RetrievalMetrics(
                num_sources=0,
                avg_source_length=0.0,
                source_diversity=0.0
            )
        
        # Calculate average source title length (proxy for document size)
        avg_length = sum(len(s.get('title', '')) for s in sources) / len(sources)
        
        # Calculate diversity: ratio of unique words to total words in titles
        all_titles = ' '.join(s.get('title', '') for s in sources).lower()
        words = re.findall(r'\w+', all_titles)
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0.0
        
        return RetrievalMetrics(
            num_sources=len(sources),
            avg_source_length=avg_length,
            source_diversity=diversity
        )
    
    @staticmethod
    def calculate_answer_metrics(
        response: str, 
        sources: List[Dict[str, Any]]
    ) -> AnswerMetrics:
        """
        Calculate metrics about the generated answer.
        
        Args:
            response: The generated answer text
            sources: List of source documents
            
        Returns:
            AnswerMetrics
        """
        # Count citations (look for patterns like "artikel", "ifølge", etc.)
        citation_patterns = [
            r'\[.*?\]',  # [source]
            r'ifølge',   # "according to" in Danish
            r'artikel',  # "article"
            r'kilde',    # "source"
        ]
        citation_count = sum(
            len(re.findall(pattern, response.lower())) 
            for pattern in citation_patterns
        )
        
        # Calculate overlap between answer and source titles
        response_words = set(re.findall(r'\w+', response.lower()))
        source_words = set()
        for source in sources:
            title_words = re.findall(r'\w+', source.get('title', '').lower())
            source_words.update(title_words)
        
        if response_words and source_words:
            overlap = len(response_words & source_words) / len(response_words)
        else:
            overlap = 0.0
        
        return AnswerMetrics(
            answer_length=len(response),
            citation_count=citation_count,
            answer_source_overlap=overlap
        )
    
    @staticmethod
    def calculate_faithfulness_metrics(
        response: str,
        sources: List[Dict[str, Any]]
    ) -> FaithfulnessMetrics:
        """
        Calculate faithfulness metrics.
        
        Args:
            response: The generated answer
            sources: List of source documents
            
        Returns:
            FaithfulnessMetrics
        """
        # Check if answer has any citations
        has_citations = bool(re.search(r'\[.*?\]|ifølge|artikel|kilde', response.lower()))
        
        # Count claims (sentences that contain verbs - rough proxy)
        sentences = re.split(r'[.!?]+', response)
        # Simple heuristic: sentences with "er", "har", "blev" etc. are claims
        claim_patterns = r'\b(er|var|har|havde|blev|bliver|kan|skal|vil)\b'
        claims_count = sum(
            1 for sent in sentences 
            if re.search(claim_patterns, sent.lower())
        )
        
        return FaithfulnessMetrics(
            has_citations=has_citations,
            claims_count=claims_count
        )
    
    @classmethod
    def evaluate_response(
        cls,
        query: str,
        workflow_id: str,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate a complete RAG response.
        
        Args:
            query: The input query
            workflow_id: ID of the workflow that generated the response
            response: The generated answer
            sources: Retrieved source documents
            
        Returns:
            EvaluationResult with all metrics
            
        Example:
            >>> result = MetricsCalculator.evaluate_response(
            ...     query="What is sne?",
            ...     workflow_id="beta_workflow_v2_hyde",
            ...     response="Sne er...",
            ...     sources=[{...}]
            ... )
            >>> print(f"Retrieved {result.retrieval.num_sources} sources")
            >>> print(f"Answer length: {result.answer.answer_length}")
        """
        return EvaluationResult(
            query=query,
            workflow_id=workflow_id,
            retrieval=cls.calculate_retrieval_metrics(sources),
            answer=cls.calculate_answer_metrics(response, sources),
            faithfulness=cls.calculate_faithfulness_metrics(response, sources),
            response=response,
            sources=sources
        )


# Comparison utilities
def compare_workflows(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compare multiple workflow results.
    
    Args:
        results: List of EvaluationResult from different workflows
        
    Returns:
        Dictionary with comparison statistics
    """
    if not results:
        return {}
    
    comparison = {
        "workflows": [r.workflow_id for r in results],
        "retrieval": {
            "avg_sources": sum(r.retrieval.num_sources for r in results) / len(results),
            "avg_diversity": sum(r.retrieval.source_diversity for r in results) / len(results),
        },
        "answer": {
            "avg_length": sum(r.answer.answer_length for r in results) / len(results),
            "avg_citations": sum(r.answer.citation_count for r in results) / len(results),
            "avg_overlap": sum(r.answer.answer_source_overlap for r in results) / len(results),
        },
        "faithfulness": {
            "pct_with_citations": sum(r.faithfulness.has_citations for r in results) / len(results) * 100,
            "avg_claims": sum(r.faithfulness.claims_count for r in results) / len(results),
        }
    }
    
    return comparison


def print_evaluation_summary(result: EvaluationResult):
    """Print a human-readable summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"Query: {result.query}")
    print(f"Workflow: {result.workflow_id}")
    print(f"{'='*60}")
    
    print(f"\n RETRIEVAL METRICS:")
    print(f"  • Sources retrieved: {result.retrieval.num_sources}")
    print(f"  • Source diversity: {result.retrieval.source_diversity:.2f}")
    
    print(f"\n ANSWER METRICS:")
    print(f"  • Answer length: {result.answer.answer_length} chars")
    print(f"  • Citations found: {result.answer.citation_count}")
    print(f"  • Answer-source overlap: {result.answer.answer_source_overlap:.2%}")
    
    print(f"\n FAITHFULNESS METRICS:")
    print(f"  • Has citations: {'Yes' if result.faithfulness.has_citations else 'No'}")
    print(f"  • Claims made: {result.faithfulness.claims_count}")
    
    print(f"\n RESPONSE:")
    print(f"  {result.response[:200]}...")
    print(f"\n{'='*60}\n")