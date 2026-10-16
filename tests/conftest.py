from __future__ import annotations

from pathlib import Path

import pytest

from agentapp.config import Settings
from agentapp.store import Store
from agentapp.tools.base import ToolContext


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.workspace = tmp_path / "workspace"
    s.max_steps = 8
    s.tool_timeout_s = 10
    s.ensure_dirs()
    return s


@pytest.fixture
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


@pytest.fixture
def ctx(settings: Settings, store: Store) -> ToolContext:
    return ToolContext(settings.workspace, store, run_id="test", timeout_s=settings.tool_timeout_s)
