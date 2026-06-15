"""E2E test for Wu Tanchang API.

Verifies startup lifespans, multi-turn chat conversations, and reset endpoints using TestClient.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


from apps.wu_tanchang_api.app import app


class FakeChatModel:
    """Mock Chat Model to avoid real API network requests during E2E test."""

    def __init__(self, **kwargs: Any) -> None:
        self.profile = {"max_input_tokens": 1000000}
        self.kwargs = kwargs

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        res = MagicMock()
        # Mock responses from LLM
        prompt_str = str(prompt)
        if "mark_material_delivered" in prompt_str:
            # If front-end assistant is generating material
            res.content = "这是您的会议准备材料... 请过目。"
        else:
            # Fallback simple conversational reply
            res.content = "您好！我是吴探长助手。请问您的品类和预算是多少？"
        return res

    def with_structured_output(self, *args: Any, **kwargs: Any) -> MagicMock:
        mock_struct = MagicMock()
        # Mock structured output for kb_extract or other tools
        mock_struct.invoke.return_value = {
            "keywords": ["烘焙", "面包", "网红店"],
            "insights": ["这是一条测试洞察一", "这是一条测试洞察二"]
        }
        return mock_struct


@pytest.fixture
def e2e_env_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Set up temporary configs and dummy files for Wu Tanchang API."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "default_model_provider": "qwen",
  "providers": {
    "qwen": {
      "api_type": "openai-compatible",
      "base_url": "https://dashscope.test/v1",
      "api_key": "fake-key",
      "main_agent_model": "qwen-flash"
    }
  }
}
""",
        encoding="utf-8",
    )
    
    # Write mock identity files
    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (tmp_path / name).write_text(f"Persona {name} content", encoding="utf-8")
        
    # Write mock kb method files
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "METHOD.md").write_text("Method content", encoding="utf-8")
    (kb_dir / "PLAYBOOK.md").write_text("Playbook content", encoding="utf-8")
    (kb_dir / "index.json").write_text("[]", encoding="utf-8")

    monkeypatch.setenv("WU_API_CONFIG", str(config_path))
    monkeypatch.setenv("WU_API_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("WU_API_WORKSPACE", str(tmp_path))
    
    # Mock model builder and checkpoints dir
    import deepagents._models
    monkeypatch.setattr(deepagents._models, "resolve_model", lambda m: m)
    monkeypatch.setattr("apps.wu_tanchang_api.agent_factory.model_builder.ChatOpenAI", FakeChatModel)
    monkeypatch.setattr("apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir", lambda: tmp_path / "runtime")

    
    # Mock the embedding retrieval for KB search if called
    monkeypatch.setattr("apps.wu_tanchang_api.agent_factory.kb_search.get_embeddings", lambda texts, *args, **kwargs: [[0.1] * 1024 for _ in texts])
    
    # Mock ChromaDB PersistentClient
    mock_chroma = MagicMock()
    monkeypatch.setattr("chromadb.PersistentClient", lambda *args, **kwargs: mock_chroma)
    
    # Mock sqlite3.connect to run against an in-memory DB or a tmp file
    db_path = tmp_path / "test_kb.db"
    monkeypatch.setenv("WU_KB_DB_PATH", str(db_path))
    
    # Initialize the test sqlite DB
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY);")
    conn.commit()
    conn.close()

    yield tmp_path


def test_wu_tanchang_api_e2e_conversation(e2e_env_setup) -> None:
    """Test E2E multi-turn conversation and reset endpoint using FastAPI TestClient."""
    # TestClient triggers FastAPI's startup event, loading state
    with TestClient(app) as client:
        # Retrieve the initialized AppState
        from apps.wu_tanchang_api.app import _state
        assert _state is not None
        
        real_agent = _state.agents["default"]
        
        # Mock ainvoke to simulate LLM updates and state persistence in the checkpointer
        async def mock_ainvoke(input_data: Any, config: Any = None) -> Any:
            messages = input_data.get("messages", [])
            last_msg = messages[-1].content if messages else ""
            
            # Get existing history
            checkpoint = await real_agent.checkpointer.aget(config) if hasattr(real_agent, "checkpointer") else None
            history = list(checkpoint.get("channel_values", {}).get("messages", [])) if checkpoint else []
            
            # Append human message and AI reply
            history.append(messages[-1])
            
            from langchain_core.messages import AIMessage
            if "你好" in last_msg:
                reply_msg = AIMessage(content="您好！我是吴探长助手。请问您的品类和预算是多少？")
            else:
                reply_msg = AIMessage(content="这是您的会议准备材料... 请过目。")
                
            history.append(reply_msg)
            
            # Save history to checkpointer
            if hasattr(real_agent, "checkpointer") and checkpoint:
                checkpoint["channel_values"]["messages"] = history
                await real_agent.checkpointer.aput(config, checkpoint, {})
                
            return {"messages": history}
            
        real_agent.ainvoke = mock_ainvoke
        
        # 1. Health check
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
        
        # 2. First turn of conversation
        chat_req = {
            "user_id": "test-user-e2e",
            "message": "你好，我想咨询餐饮开店",
            "conversation_id": "test-conv-e2e"
        }
        res = client.post("/chat", json=chat_req)
        assert res.status_code == 200
        data = res.json()
        assert "user_id" in data
        assert data["conversation_id"] == "test-conv-e2e"
        assert "reply" in data
        assert "您好！我是吴探长助手" in data["reply"]
        
        # 3. Second turn of conversation (multi-turn)
        chat_req_2 = {
            "user_id": "test-user-e2e",
            "message": "我的品类是烘焙，预算在20万左右",
            "conversation_id": "test-conv-e2e"
        }
        res_2 = client.post("/chat", json=chat_req_2)
        assert res_2.status_code == 200
        data_2 = res_2.json()
        assert data_2["conversation_id"] == "test-conv-e2e"
        assert "这是您的会议准备材料" in data_2["reply"]
        
        # 4. Reset conversation
        reset_req = {
            "user_id": "test-user-e2e",
            "conversation_id": "test-conv-e2e"
        }
        res_reset = client.post("/reset", json=reset_req)
        assert res_reset.status_code == 200
        assert res_reset.json()["ok"] is True


@pytest.mark.skipif(
    os.environ.get("WU_E2E_REAL_LLM") not in {"1", "true", "TRUE", "yes", "YES"},
    reason="Set WU_E2E_REAL_LLM=1 to run E2E test with real LLM"
)
def test_wu_tanchang_api_e2e_real_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """E2E test with real LLM model and real workspace."""
    # 1. Point to real configuration and workspace
    project_root = Path(__file__).resolve().parents[1]
    api_dir = project_root / "apps" / "wu_tanchang_api"

    config_path = api_dir / "config.json"
    env_path = api_dir / ".env"
    workspace_path = api_dir / "workspace"

    # We create a temporary runtime directory inside tmp_path so we don't pollute the real one
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("WU_API_CONFIG", str(config_path))
    monkeypatch.setenv("WU_API_ENV_FILE", str(env_path))
    monkeypatch.setenv("WU_API_WORKSPACE", str(workspace_path))
    monkeypatch.setattr("apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir", lambda: runtime_dir)

    # Do NOT mock resolve_model, ChatOpenAI, or get_embeddings. Let them load naturally.
    from apps.wu_tanchang_api.config import load_env_file
    load_env_file(env_path)

    # Ensure DASHSCOPE_API_KEY is populated for the langchain SDK/provider client
    dashscope_key = os.environ.get("WU_DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope_key:
        pytest.fail("WU_DASHSCOPE_API_KEY or DASHSCOPE_API_KEY is required to run the real LLM test.")

    if dashscope_key and not os.environ.get("DASHSCOPE_API_KEY"):
        monkeypatch.setenv("DASHSCOPE_API_KEY", dashscope_key)

    # Re-create the FastAPI app so it initializes the AppState using the new environment configuration
    from apps.wu_tanchang_api.app import create_app
    real_app = create_app()

    with TestClient(real_app) as client:
        # 1. Health check
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # 2. First turn of conversation
        chat_req = {
            "user_id": "real-user-e2e",
            "message": "你好，我想咨询餐饮开店",
            "conversation_id": "real-conv-e2e"
        }
        res = client.post("/chat", json=chat_req)
        assert res.status_code == 200
        data = res.json()
        assert "user_id" in data
        assert data["conversation_id"] == "real-conv-e2e"
        assert "reply" in data
        assert len(data["reply"].strip()) > 0
        print("\n[Real LLM Reply 1]:", data["reply"])

        # 3. Second turn of conversation (multi-turn)
        chat_req_2 = {
            "user_id": "real-user-e2e",
            "message": "我的品类是烘焙，预算在20万左右",
            "conversation_id": "real-conv-e2e"
        }
        res_2 = client.post("/chat", json=chat_req_2)
        assert res_2.status_code == 200
        data_2 = res_2.json()
        assert data_2["conversation_id"] == "real-conv-e2e"
        assert "reply" in data_2
        assert len(data_2["reply"].strip()) > 0
        print("\n[Real LLM Reply 2]:", data_2["reply"])

        # 4. Reset conversation
        reset_req = {
            "user_id": "real-user-e2e",
            "conversation_id": "real-conv-e2e"
        }
        res_reset = client.post("/reset", json=reset_req)
        assert res_reset.status_code == 200
        assert res_reset.json()["ok"] is True


