"""pytest configuration.

Only test_dual_mode_simple.py runs offline. The other scripts import the full agent, which
needs ANTHROPIC_API_KEY (anthropic_engine exits without it) and downloads the
all-MiniLM-L6-v2 embedding model; several are interactive. They are skipped unless a key is set.
Run them from the repository root, e.g. `python tests/test_rag.py`.
"""
import os

collect_ignore = []
if not os.environ.get("ANTHROPIC_API_KEY"):
    collect_ignore = [
        "test_agent_rag.py",
        "test_dual_mode.py",
        "test_embedding_on_exit.py",
        "test_rag.py",
        "test_rag_automated.py",
    ]
