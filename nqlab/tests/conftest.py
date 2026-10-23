from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pytest

from nqlab.settings import clear_settings_cache, get_settings

REPO = Path(__file__).resolve().parents[1]


def make_root(tmp: Path) -> Path:
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    for f in (REPO / "config").glob("*.yaml"):
        shutil.copy(f, tmp / "config" / f.name)
    for d in ("data/raw", "data/parquet", "data/holdout", "research_log/llm", "research_log/hypotheses", "specs", "reports", "results"):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    for f in (REPO / "specs").glob("*.yaml"):
        shutil.copy(f, tmp / "specs" / f.name)
    return tmp


@pytest.fixture()
def root(tmp_path, monkeypatch) -> Path:
    r = make_root(tmp_path / "proj")
    monkeypatch.setenv("NQLAB_ROOT", str(r))
    clear_settings_cache()
    yield r
    clear_settings_cache()


@pytest.fixture()
def settings(root):
    return get_settings(root)


@pytest.fixture(scope="session")
def session_settings(tmp_path_factory):
    r = make_root(tmp_path_factory.mktemp("sess_proj"))
    return get_settings(r)


@pytest.fixture(scope="session")
def small_md(session_settings):
    """~3 months of synthetic random-walk data (full Globex sessions)."""
    from nqlab.data.market import MarketData
    from nqlab.data.synthetic import random_walk_bars

    raw = random_walk_bars(session_settings, dt.date(2024, 1, 2), dt.date(2024, 3, 29), seed=7)
    return MarketData.from_raw(raw, session_settings, data_version="small")
