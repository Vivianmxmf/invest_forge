"""FakeDataProvider tests."""
from __future__ import annotations

import pytest

from janus_terminal.tools.data_tools import FakeDataProvider, fetch_for_codes


@pytest.mark.unit
def test_fundamentals_from_sample(fake_data: FakeDataProvider):
    fm = fake_data.fundamentals("TEST.SH")
    assert fm.ts_code == "TEST.SH"
    assert fm.revenue == pytest.approx(1.2e10)
    assert 0 < fm.net_margin < 1
    assert fm.pe_ratio == pytest.approx(18.0)


@pytest.mark.unit
def test_fundamentals_falls_back_to_default(fake_data: FakeDataProvider):
    fm = fake_data.fundamentals("UNKNOWN.SH")
    # falls back to the bundled DEFAULT (revenue=4.32e10)
    assert fm.ts_code == "UNKNOWN.SH"
    assert fm.revenue == pytest.approx(4.32e10)


@pytest.mark.unit
def test_news_limits(fake_data: FakeDataProvider):
    items = fake_data.news("TEST.SH", limit=1)
    assert len(items) == 1
    assert items[0].title.startswith("Q1")


@pytest.mark.unit
def test_macro_default(fake_data: FakeDataProvider):
    macro = fake_data.macro()
    assert macro.pmi == pytest.approx(51.2)


@pytest.mark.unit
def test_fetch_for_codes(fake_data: FakeDataProvider):
    out = fetch_for_codes(fake_data, ["TEST.SH", "UNKNOWN.SZ"])
    assert {"TEST.SH", "UNKNOWN.SZ"} == set(out.keys())


@pytest.mark.unit
def test_composite_provider_falls_back_for_macro():
    """Regression: TushareProvider.macro() raises; the composite must catch
    NotImplementedError and fall back to the fake macro source.
    (Codex review round-1 P1.)
    """
    from janus_terminal.common.types import FinancialMetrics, MacroContext
    from janus_terminal.tools.data_tools import CompositeProvider

    class FakePrimary:
        name = "primary"

        def fundamentals(self, ts_code):
            return FinancialMetrics(
                ts_code=ts_code, fiscal_year=2024, revenue=1.0, net_profit=1.0,
                gross_margin=0.2, net_margin=0.1, roe=0.1, roic=0.1,
                debt_ratio=0.3, free_cash_flow=1.0,
            )

        def news(self, ts_code, *, limit=20):
            return []

        def macro(self):
            raise NotImplementedError("primary has no macro endpoint")

    composite = CompositeProvider(fundamentals_news=FakePrimary(), macro_provider=FakePrimary())
    macro = composite.macro()
    assert isinstance(macro, MacroContext)
    assert macro.pmi  # fell back to FakeDataProvider defaults


@pytest.mark.unit
def test_sample_fundamentals_match_kb_excerpt():
    """Sample revenue must match the KB markdown ('432 亿元' = 4.32e10).
    (Codex review round-1 P2.)"""
    from janus_terminal.common.config import get_settings

    settings = get_settings()
    if not (settings.paths.sample_dir / "fundamentals.json").exists():
        pytest.skip("sample dataset not generated (run scripts/generate_sample_data.py)")

    p = FakeDataProvider(sample_dir=settings.paths.sample_dir)
    smic = p.fundamentals("688981.SH")
    assert smic.revenue == pytest.approx(43.2e9, rel=0.01)
    mutai = p.fundamentals("600519.SH")
    assert mutai.revenue == pytest.approx(147e9, rel=0.01)
