"""Test the metrics with your connector."""

import asyncio
from lex_eval.connectors.lex_llm_connector import LexLLMConnector
from lex_eval.metrics.metrics import MetricsCalculator, print_evaluation_summary

user_query = "Hvad er Aasiaat og hvad er byens vigtigste erhverv?"
workflow = "beta_workflow_v2_hyde"

async def test_metrics():
    """Test metrics on a real workflow response."""
    
    connector = LexLLMConnector()
    
    # Run a workflow
    result = await connector.run_workflow(
        workflow_id=workflow,
        user_input=user_query,
        conversation_id="metrics-test"
    )
    
    # Calculate metrics
    eval_result = MetricsCalculator.evaluate_response(
        query=user_query,
        workflow_id=workflow,
        response=result.response,
        sources=[s.model_dump() for s in result.sources]
    )
    
    # Print summary
    print_evaluation_summary(eval_result)


if __name__ == "__main__":
    asyncio.run(test_metrics())
