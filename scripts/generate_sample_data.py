#!/usr/bin/env python3
"""Generate deterministic synthetic sample data for InvestForge.

Outputs (under ``data/sample/``):
  * ``fundamentals.json`` — 5 stocks across 5 sectors with 3 fiscal years each
  * ``news.json``          — 5 headlines per stock with sentiment scores
  * ``macro.json``         — single MacroContext snapshot
  * ``kb/`` directory      — 5 markdown "annual report excerpts" for RAG
  * ``prices.csv``         — daily OHLCV synthetic series (1 year × 5 stocks)
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "kb").mkdir(parents=True, exist_ok=True)

SEED = 42

STOCKS = [
    # (ts_code, name, sector, revenue_bn, net_margin, growth)
    ("688981.SH", "中芯国际", "半导体", 43.2, 0.151, 0.22),
    ("600519.SH", "贵州茅台", "白酒/消费", 147.0, 0.523, 0.15),
    ("300750.SZ", "宁德时代", "新能源", 360.0, 0.092, 0.18),
    ("600276.SH", "恒瑞医药", "医药", 22.8, 0.183, 0.06),
    ("601398.SH", "工商银行", "银行", 870.0, 0.412, 0.03),
]


def fundamentals_json() -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    out: dict[str, dict] = {}
    for code, _name, _sector, rev, nm, growth in STOCKS:
        out[code] = {
            "fiscal_year": 2024,
            "revenue": rev * 1e9,
            "net_profit": rev * nm * 1e9,
            "gross_margin": float(rng.uniform(0.20, 0.55)),
            "net_margin": nm,
            "roe": float(rng.uniform(0.08, 0.25)),
            "roic": float(rng.uniform(0.06, 0.20)),
            "debt_ratio": float(rng.uniform(0.25, 0.55)),
            "free_cash_flow": rev * nm * 0.85 * 1e9,
            "pe_ratio": float(rng.uniform(12.0, 35.0)),
            "pb_ratio": float(rng.uniform(1.2, 5.5)),
            "market_cap": rev * nm * 22.0 * 1e9,
        }
    return out


def news_json() -> dict[str, list[dict]]:
    """Five headlines per stock; sentiment annotated."""
    rng = np.random.default_rng(SEED + 1)
    headlines = {
        "688981.SH": [
            ("公司发布 2024 年报, 净利润同比 +28%", "positive", 0.42),
            ("12 寸晶圆产能持续扩张, 国产替代进度领先", "positive", 0.30),
            ("美国出口管制再升级, 高端工艺面临压力", "negative", -0.35),
            ("机构调研活跃, 公募基金二季度增持", "positive", 0.18),
            ("一季度业绩低于市场预期, 短期承压", "negative", -0.22),
        ],
        "600519.SH": [
            ("茅台一季度营收增长稳健, 高端白酒韧性凸显", "positive", 0.31),
            ("公司宣布提价, 渠道反响积极", "positive", 0.40),
            ("消费税改革预期升温, 板块情绪波动", "negative", -0.20),
            ("分红比例创新高, 现金奶牛属性强化", "positive", 0.36),
            ("二三线白酒挤压终端, 行业竞争加剧", "negative", -0.18),
        ],
        "300750.SZ": [
            ("公司拿下欧洲大单, 海外营收占比突破 30%", "positive", 0.45),
            ("新一代麒麟电池量产, 能量密度行业领先", "positive", 0.38),
            ("锂价下行压制毛利, 一季度盈利同比承压", "negative", -0.28),
            ("固态电池进度低于市场预期, 估值下修", "negative", -0.22),
            ("储能业务订单饱满, 排产至明年中", "positive", 0.27),
        ],
        "600276.SH": [
            ("创新药管线进度顺利, 多款新药获批", "positive", 0.34),
            ("集采新一轮谈判落地, 仿制药降价压力延续", "negative", -0.30),
            ("一季度研发投入再创新高, 占营收 18%", "positive", 0.20),
            ("海外授权交易突破, BD 战略转型见效", "positive", 0.32),
            ("地缘政治不确定性, 美元资产估值承压", "negative", -0.18),
        ],
        "601398.SH": [
            ("国有大行净息差企稳, 资产质量稳健", "positive", 0.24),
            ("分红率维持 30%+, 股息率吸引力提升", "positive", 0.30),
            ("房贷利率下行, 短期息差仍承压", "negative", -0.18),
            ("地方债置换推进, 资产结构改善", "positive", 0.22),
            ("零售业务转型加速, 财富管理 AUM 突破 2 万亿", "positive", 0.26),
        ],
    }
    out: dict[str, list[dict]] = {}
    base = date(2025, 3, 1)
    for code, items in headlines.items():
        out[code] = []
        for i, (title, label, sentiment) in enumerate(items):
            out[code].append(
                {
                    "title": title,
                    "published": (base + timedelta(days=int(rng.integers(0, 60)))).isoformat(),
                    "source": "sample",
                    "sentiment": sentiment,
                    "label": label,
                    "summary": title,
                }
            )
    return out


def macro_json() -> dict:
    return {
        "cpi_yoy": 0.4,
        "ppi_yoy": -2.7,
        "m1_yoy": 1.2,
        "m2_yoy": 7.0,
        "pmi": 50.4,
        "us10y_yield": 4.21,
        "cny_dxy": 104.3,
        "summary": (
            "国内 CPI 温和回升, PPI 仍处通缩区间; 制造业 PMI 50.4 重回扩张; "
            "海外美元利率高位运行, 北向资金边际偏弱. 整体宏观偏防御, 高 ROE / "
            "高股息板块更受青睐."
        ),
    }


def kb_markdown() -> dict[str, str]:
    """Synthetic 'annual report excerpts' that the RAG retriever can hit."""
    return {
        "688981_excerpt.md": (
            "# 中芯国际 2024 年报摘要\n\n"
            "公司 2024 年实现营业收入 432 亿元, 同比 +22%; 归母净利润 65 亿元, 同比 +28%; "
            "综合毛利率 28.5%, 净利率 15.1%; 经营性现金流净额 89 亿元.\n\n"
            "## 风险提示\n"
            "1. 客户集中度较高, 前五大客户占比 60%;\n"
            "2. 海外销售敞口暴露汇率波动;\n"
            "3. 美国出口管制可能影响 7nm 以下工艺设备的引入."
        ),
        "600519_excerpt.md": (
            "# 贵州茅台 2024 年报摘要\n\n"
            "公司 2024 年实现营业收入 1470 亿元, 同比 +15%; 归母净利润 770 亿元, 同比 +18%; "
            "毛利率 92%, 净利率 52.3%; 经营性现金流净额 850 亿元.\n\n"
            "## 风险提示\n"
            "1. 渠道价格管控压力;\n"
            "2. 消费税改革预期;\n"
            "3. 高端白酒消费场景受经济周期影响."
        ),
        "300750_excerpt.md": (
            "# 宁德时代 2024 年报摘要\n\n"
            "公司 2024 年实现营业收入 3600 亿元, 同比 +18%; 归母净利润 332 亿元, 同比 +9%; "
            "毛利率 23%, 净利率 9.2%; 经营性现金流净额 420 亿元.\n\n"
            "## 风险提示\n"
            "1. 锂价波动对毛利率扰动;\n"
            "2. 海外建厂周期长, 资本开支压力;\n"
            "3. 固态电池产业化进度不及预期."
        ),
        "600276_excerpt.md": (
            "# 恒瑞医药 2024 年报摘要\n\n"
            "公司 2024 年实现营业收入 228 亿元, 同比 +6%; 归母净利润 42 亿元, 同比 +5%; "
            "毛利率 84%, 净利率 18.3%; R&D 投入占比 18%.\n\n"
            "## 风险提示\n"
            "1. 集采新一轮谈判降价压力;\n"
            "2. 创新药出海进度;\n"
            "3. 美元资产估值波动."
        ),
        "601398_excerpt.md": (
            "# 工商银行 2024 年报摘要\n\n"
            "公司 2024 年实现营业收入 8700 亿元, 同比 +3%; 归母净利润 3590 亿元, 同比 +3%; "
            "净息差 1.45%, 不良贷款率 1.36%; 分红比例 30%+.\n\n"
            "## 风险提示\n"
            "1. 净息差仍处低位;\n"
            "2. 地产业风险敞口;\n"
            "3. 房贷利率下行压制零售盈利."
        ),
    }


def prices_csv() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    days = pd.date_range("2024-01-02", "2024-12-31", freq="B")  # business days
    frames: list[pd.DataFrame] = []
    for code, *_ in STOCKS:
        rets = rng.normal(0.0005, 0.018, len(days))
        prices = 50 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "trade_date": days,
                "ts_code": code,
                "open": prices * (1 + rng.normal(0, 0.003, len(days))),
                "high": prices * (1 + np.abs(rng.normal(0, 0.006, len(days)))),
                "low": prices * (1 - np.abs(rng.normal(0, 0.006, len(days)))),
                "close": prices,
                "volume": rng.integers(1_000_000, 50_000_000, len(days)),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    (OUT / "fundamentals.json").write_text(
        json.dumps(fundamentals_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "news.json").write_text(
        json.dumps(news_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "macro.json").write_text(
        json.dumps(macro_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, body in kb_markdown().items():
        (OUT / "kb" / name).write_text(body, encoding="utf-8")
    prices = prices_csv()
    prices.to_csv(OUT / "prices.csv", index=False)

    size_total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(
        f"wrote sample dataset under {OUT.relative_to(ROOT)} "
        f"({size_total / 1024:.1f} KB)"
    )
    print(f"  - {len(STOCKS)} stocks × 1 fiscal year fundamentals")
    print(f"  - {sum(len(v) for v in news_json().values())} news items")
    print(f"  - {len(prices)} price rows ({prices['ts_code'].nunique()} stocks)")
    print(f"  - {len(list((OUT / 'kb').iterdir()))} KB documents")


if __name__ == "__main__":
    sys.exit(main())
