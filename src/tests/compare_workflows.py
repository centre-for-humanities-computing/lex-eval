"""
Compare multiple workflows on the same query.
"""

import asyncio
from lex_eval.connectors.lex_llm_connector import LexLLMConnector
from lex_eval.metrics.metrics import MetricsCalculator, print_evaluation_summary, compare_workflows
import json


async def compare_workflows_on_query(
    connector: LexLLMConnector,
    query: str,
    workflow_ids: list[str]
):
    """
    Run the same query on multiple workflows and compare results.
    
    Args:
        connector: LexLLMConnector instance
        query: The query to test
        workflow_ids: List of workflow IDs to compare
    """
    print(f"\n{'='*70}")
    print(f"COMPARING WORKFLOWS ON QUERY: {query}")
    print(f"{'='*70}\n")
    
    results = []
    
    for workflow_id in workflow_ids:
        print(f"Running workflow: {workflow_id}...")
        
        try:
            # Run workflow
            workflow_result = await connector.run_workflow(
                workflow_id=workflow_id,
                user_input=query,
                conversation_id=f"compare-{workflow_id}"
            )
            
            # Calculate metrics
            eval_result = MetricsCalculator.evaluate_response(
                query=query,
                workflow_id=workflow_id,
                response=workflow_result.response,
                sources=[s.model_dump() for s in workflow_result.sources]
            )
            
            results.append(eval_result)
            
            # Print individual result
            print_evaluation_summary(eval_result)
            
        except Exception as e:
            print(f" Failed to run {workflow_id}: {e}\n")
    
    # Print comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        comparison = compare_workflows(results)
        
        print(f"Workflows compared: {', '.join(comparison['workflows'])}\n")
        
        print(" RETRIEVAL:")
        print(f"  • Avg sources: {comparison['retrieval']['avg_sources']:.1f}")
        print(f"  • Avg diversity: {comparison['retrieval']['avg_diversity']:.2f}")
        
        print("\n ANSWER QUALITY:")
        print(f"  • Avg length: {comparison['answer']['avg_length']:.0f} chars")
        print(f"  • Avg citations: {comparison['answer']['avg_citations']:.1f}")
        print(f"  • Avg overlap: {comparison['answer']['avg_overlap']:.2%}")
        
        print("\n FAITHFULNESS:")
        print(f"  • With citations: {comparison['faithfulness']['pct_with_citations']:.0f}%")
        print(f"  • Avg claims: {comparison['faithfulness']['avg_claims']:.1f}")
        
        # Determine best workflow by composite score
        print(f"\n{'='*70}")
        print("RANKING (by composite score)")
        print(f"{'='*70}\n")
        
        ranked = []
        for result in results:
            # Simple composite score (customize weights as needed)
            score = (
                result.retrieval.num_sources * 0.2 +
                result.retrieval.source_diversity * 0.2 +
                result.answer.citation_count * 0.3 +
                result.answer.answer_source_overlap * 0.3
            )
            ranked.append((result.workflow_id, score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        for i, (wf_id, score) in enumerate(ranked, 1):
            print(f"{i}. {wf_id}: {score:.3f}")
    
    return results


async def main():
    """Main comparison script."""
    
    connector = LexLLMConnector()
    
    # Test query
    query = "Hvad er Aasiaat og hvad er byens vigtigste erhverv?"
    
    # Workflows to compare
    workflows = [
        "beta_workflow_v2_hyde",
        "beta_workflow_v2_hybrid",
        "beta_workflow_v2_hybrid_hyde",
    ]
    
    # Run comparison
    results = await compare_workflows_on_query(connector, query, workflows)
    
    # Optionally save results to JSON
    results_json = [r.model_dump() for r in results]
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n Results saved to comparison_results.json")


if __name__ == "__main__":
    asyncio.run(main())