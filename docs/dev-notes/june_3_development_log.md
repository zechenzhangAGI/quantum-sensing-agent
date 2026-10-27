# Development Log - June 3, 2024 (Simulated Date)

This log details the work done to address the items in `todo.md` for the 'June 3rd' (originally June 2nd in issue) tasks.

## Changes Implemented:

### 1. Modified Embedding Storage for Chunking
*   **File Modified:** `agent/agent.py`
*   **Changes:**
    *   Added `self.embedding_chunk_size = 10` to `NVExperimentAgent.__init__`.
    *   Updated the `save_conversation_embeddings` method to save conversation history in chunks of 10 turns.
    *   Each chunk is saved as a separate JSON file (e.g., `conversation_chunk_X_timestamp.json`) in the `embeddings_dir`.
    *   Each embedding chunk now includes metadata about plots analyzed *within that specific chunk*.
*   **Reasoning:** This addresses the requirement to move away from embedding the entire conversation history as one file, allowing for more granular RAG retrieval and potentially better management of embedding data.

### 2. Implemented RAG as a Tool
*   **File Modified:** `agent/agent.py`
*   **Changes:**
    *   Added a new `rag_search` action type to the agent's capabilities.
    *   The `system_instruction` was updated to inform the LLM about this new tool, how to use it (with a query in the `content` field), and when (e.g., if stuck).
    *   A new method `_action_rag_search(self, query: str)` was implemented to:
        *   Take a query string.
        *   Call `_get_rag_context(query)` to search through existing (now chunked) embeddings.
        *   Add the formatted RAG results to the conversation history as an assistant message, making them available to the LLM.
    *   The `handle_user_input` method was updated to call `_action_rag_search` when this action is invoked. No explicit user permission is asked for this action as it's an internal information retrieval mechanism.
*   **Reasoning:** This fulfills the requirement to allow the agent to actively use RAG as a tool when it determines a need to search its past experiences, rather than only relying on the automatic RAG triggered by new user messages. The query for this tool will be formulated by the LLM itself based on its current context.

### 3. Implemented Anthropic Cached Input
*   **File Modified:** `agent/anthropic_engine.py`
*   **Changes:**
    *   Modified the `call_llm` function to enable prompt caching for the `system_message`.
    *   The `system_message` (a string) is now wrapped into a list of content blocks, with the main text block having `{"cache_control": {"type": "ephemeral"}}` added.
    *   This leverages Anthropic's caching mechanism to potentially reduce costs and latency by caching the static, lengthy system prompt.
*   **Reasoning:** Addresses the task to investigate and implement Anthropic's cached input feature to save money. The default 5-minute ephemeral cache was used.

## Future Tips & Observations:

*   **RAG Query Strategy for Tool:** The current `rag_search` tool relies on the LLM to formulate a good query. Future improvements could involve providing the LLM with more guidance or structured ways to formulate queries for the RAG tool, or even having predefined query templates for common situations where it might get stuck.
*   **RAG Chunking in `rag_engine.py`:** The `rag_engine.search_similar` function currently splits text from an embedding file (which is already a chunk of conversation) into further sub-chunks by newline. This means the search is effectively happening on sub-sub-chunks. This might be too granular. Consider revising `search_similar` to treat each loaded embedding file (i.e., each conversation chunk) as a single document for similarity search against the query, rather than splitting it further. This would require embedding the entire content of `full_chunk_text` (from `agent.py`'s `save_conversation_embeddings`) as one vector in `rag_engine.save_embeddings`, and then `search_similar` would compare the query vector against these single chunk vectors.
*   **Anthropic Caching for Conversation History:** While the system prompt is now cached, `cache_control` could also be applied to parts of the `messages` array in `anthropic_engine.py` to cache parts of the conversation history. This could be beneficial for very long conversations, though it adds complexity in managing which parts of the history are marked for caching.
*   **Testing Environment:** The `todo.md` mentions using `agent_mac.py` for testing. The changes made are primarily in `agent.py` and `anthropic_engine.py`. Thorough testing in an environment that can execute these (like the one `agent_mac.py` is intended for) is crucial. Test scripts like `test_agent_rag.py` should also be reviewed and potentially updated.
*   **1-Hour Cache TTL:** For the system prompt, the 5-minute cache is likely fine as it's used frequently. However, if chunked conversation embeddings are found to be useful beyond 5 minutes but within an hour, the 1-hour cache TTL (using the `extended-cache-ttl-2025-04-11` beta header and `"ttl": "1h"` in `cache_control`) could be explored for those.
*   **Error Handling in RAG:** The RAG tool implementation currently adds results (or a 'not found' message) to history. More sophisticated error handling or feedback mechanisms could be added if RAG searches consistently fail or return poor results.
*   **Configuration for Chunk Size:** The `embedding_chunk_size` is hardcoded to 10. This could be made configurable, perhaps via an environment variable or a settings file, if different chunk sizes need to be experimented with.
