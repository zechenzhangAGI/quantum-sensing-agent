"""Offline test of NVExperimentAgent.save_conversation_embeddings (no API key, no model download).

The LLM and embedding modules are replaced by stubs, so only the chunking and saving logic runs.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))


def _import_agent_with_stubs(saved):
    llm = types.ModuleType("anthropic_engine")
    llm.call_llm = llm.call_vision = lambda *a, **k: ""
    rag = types.ModuleType("rag_engine")
    rag.embed_text = rag.load_embeddings = rag.search_similar = lambda *a, **k: None
    rag.save_embeddings = lambda text, path: saved.append((text, path))
    sys.modules["anthropic_engine"] = llm
    sys.modules["rag_engine"] = rag
    sys.modules.pop("agent", None)
    import agent
    return agent


def test_save_conversation_embeddings(tmp_path):
    saved = []
    agent = _import_agent_with_stubs(saved)
    a = agent.NVExperimentAgent.__new__(agent.NVExperimentAgent)
    a.embedding_chunk_size, a.embedding_chunk_overlap = 10, 5
    a.embeddings_dir = str(tmp_path)
    a.logfile_path = str(tmp_path / "log.txt")
    a._get_available_plots = lambda: []
    a.conversation_history = [{"role": "user" if i % 2 else "assistant", "content": f"turn {i}"} for i in range(12)]
    a.conversation_history.append({"role": "rag_search_result", "content": "retrieved text"})

    a.save_conversation_embeddings()

    assert len(saved) == 2  # 12 turns, chunks of 10 with step 5 -> turns 0-9 and 5-11
    assert all("retrieved text" not in text for text, _ in saved)
    assert all(os.path.dirname(p) == str(tmp_path) for _, p in saved)
