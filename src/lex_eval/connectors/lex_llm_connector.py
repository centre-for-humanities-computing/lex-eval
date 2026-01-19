"""
Connector for communicating with the lex-llm API service.
This allows lex-eval to run workflows .
"""

from typing import List, Dict, Any, Optional
import httpx
import os
import json
from pydantic import BaseModel


# Models matching lex-llm's response format
class Source(BaseModel):
    """A source document returned by a workflow."""
    id: int | str
    title: str
    url: str


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    role: str  # "system", "user", "assistant"
    content: str


class WorkflowResult(BaseModel):
    """Complete result from running a workflow."""
    conversation_id: str
    run_id: str
    response: str  # The final LLM response
    sources: List[Source]  # Retrieved documents
    conversation_history: List[ConversationMessage]  # Updated history


class WorkflowMetadata(BaseModel):
    """Metadata about a workflow."""
    workflow_id: str
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: Optional[List[str]] = None


class LexLLMConnector:
    """Handles communication with the lex-llm service."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the connector.
        
        Args:
            base_url: The base URL of the lex-llm service. 
                    Defaults to LEX_LLM_HOST env var or http://localhost:8001
        """
        self.base_url = base_url or os.getenv("LEX_LLM_HOST", "http://localhost:8001")
        self.timeout = httpx.Timeout(300.0, connect=10.0)  # 5 min for long workflows
    
    async def run_workflow(
        self,
        workflow_id: str,
        user_input: str,
        conversation_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> WorkflowResult:
        """
        Run a workflow and return the complete result.
        
        Args:
            workflow_id: ID of the workflow to run (e.g., "beta_workflow_v2_hyde")
            user_input: The user's query/input
            conversation_id: Unique conversation identifier
            conversation_history: Previous messages in the conversation
            
        Returns:
            WorkflowResult containing response, sources, and updated history
            
        Example:
            >>> connector = LexLLMConnector()
            >>> result = await connector.run_workflow(
            ...     workflow_id="beta_workflow_v2_hyde",
            ...     user_input="What is sne?",
            ...     conversation_id="eval-123"
            ... )
            >>> print(result.response)
            >>> print(result.sources)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/workflows/{workflow_id}/run",
                    json={
                        "user_input": user_input,
                        "conversation_id": conversation_id,
                        "conversation_history": conversation_history or [],
                    },
                )
                response.raise_for_status()
                
                # Parse NDJSON streaming response
                result_data = self._parse_ndjson_stream(response.text)
                return WorkflowResult(**result_data)
                
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to lex-llm at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Workflow execution failed: {e.response.text}")
    
    def _parse_ndjson_stream(self, ndjson_text: str) -> Dict[str, Any]:
        """
        Parse NDJSON stream and extract the final result.
        
        The stream contains events like:
        - stream_start
        - workflow_step
        - sources (contains retrieved documents)
        - text_chunk (streaming response)
        - stream_end (contains final conversation_history)
        """
        response_chunks = []
        sources = []
        conversation_history = []
        conversation_id = ""
        run_id = ""
        
        for line in ndjson_text.strip().split('\n'):
            if not line:
                continue
                
            event = json.loads(line)
            event_type = event.get("event")
            
            if event_type == "stream_start":
                conversation_id = event.get("conversation_id", "")
                run_id = event.get("run_id", "")
            
            elif event_type == "sources":
                # Extract sources from the event
                sources = event.get("data", [])
            
            elif event_type == "text_chunk":
                # Accumulate response text
                chunk = event.get("data", "")
                response_chunks.append(chunk)
            
            elif event_type == "stream_end":
                # Get final conversation history
                conversation_history = event.get("data", {}).get("conversation_history", [])
        
        return {
            "conversation_id": conversation_id,
            "run_id": run_id,
            "response": "".join(response_chunks),
            "sources": sources,
            "conversation_history": conversation_history,
        }
    
    async def get_workflow_metadata(self, workflow_id: str) -> WorkflowMetadata:
        """
        Get metadata about a specific workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            WorkflowMetadata containing workflow information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/workflows/{workflow_id}/metadata"
                )
                response.raise_for_status()
                return WorkflowMetadata(**response.json())
                
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to lex-llm at {self.base_url}: {e}")
    
    async def list_workflows(self) -> List[WorkflowMetadata]:
        """
        List all available workflows.
        
        Returns:
            List of WorkflowMetadata for all available workflows
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/workflows/metadata")
                response.raise_for_status()
                
                workflows_data = response.json()
                return [WorkflowMetadata(**wf) for wf in workflows_data]
                
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to lex-llm at {self.base_url}: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if the lex-llm service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False