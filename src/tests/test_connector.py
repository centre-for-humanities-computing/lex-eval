"""
Simple test script to verify the LexLLMConnector works.
"""

import asyncio
from lex_eval.connectors.lex_llm_connector import LexLLMConnector


async def test_connector():
    """Test the connector against a running lex-llm instance."""
    
    # Initialize connector
    connector = LexLLMConnector()
    
    print("=" * 60)
    print("Testing LexLLMConnector")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    is_healthy = await connector.health_check()
    if is_healthy:
        print("    lex-llm service is healthy")
    else:
        print("    lex-llm service is not reachable")
        return
    
    # Test 2: List workflows
    print("\n2. Listing available workflows...")
    try:
        workflows = await connector.list_workflows()
        print(f"   Found {len(workflows)} workflows:")
        for wf in workflows:
            print(f"   - {wf.workflow_id}: {wf.name}")
    except Exception as e:
        print(f"    Failed to list workflows: {e}")
        return
    
    # Test 3: Get specific workflow metadata
    print("\n3. Getting workflow metadata...")
    try:
        metadata = await connector.get_workflow_metadata("beta_workflow_v2_hyde")
        print(f"    Workflow: {metadata.name}")
        print(f"     Description: {metadata.description}")
        print(f"     Tags: {metadata.tags}")
    except Exception as e:
        print(f"    Failed to get metadata: {e}")
    
    # Test 4: Run a workflow
    print("\n4. Running workflow with test query...")
    try:
        result = await connector.run_workflow(
            workflow_id="beta_workflow_v2_hyde",
            user_input="Hvad er Aasiaat og hvad er byens vigtigste erhverv?",
            conversation_id="test-123"
        )
        
        print(f"   Workflow completed!")
        print(f"   Conversation ID: {result.conversation_id}")
        print(f"   Run ID: {result.run_id}")
        print(f"   Response length: {len(result.response)} characters")
        print(f"   Number of sources: {len(result.sources)}")
        
        if result.sources:
            print(f"   Sources retrieved:")
            for source in result.sources:
                print(f"   - [{source.id}] {source.title}")
                print(f"     URL: {source.url}")
        
        print(f"\n   Response preview:")
        print(f"   {result.response[:200]}...")
        
    except Exception as e:
        print(f"    Failed to run workflow: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_connector())
