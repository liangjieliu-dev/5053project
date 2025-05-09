"""
从 2020‑01‑01 起：
1. 下载 ^GSPC、GLD、USO、ACWI、^IRX 的调整后收盘价
2. 计算日对数收益率
3. 构造 50% GLD + 50% USO 组合
4. 输出：
   • returns_daily.csv        —— 每日对数收益率
   • desc_stats.csv          —— 描述统计（含 skew、kurtosis）
   • 在控制台打印 Beta、Sharpe、Treynor、Jensen α
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- 1. 下载数据 ----------
START = "2020-12-30"
END   = "2025-05-01"
TICKERS = ["^GSPC", "GLD", "USO", "ACWI", "^IRX"]

data = yf.download(TICKERS, start=START, end=END, auto_adjust=True)

# ---------- 2. 处理价格并计算对数收益 ----------
# 2.1 取调整后价格并删除缺失行，避免 log 报错
adj = data["Close"]

# 把 ≤0 的价格设为 NaN
adj = adj.mask(adj <= 0, np.nan)

# ② 删除：本日或前一日有 NaN 的行
adj = adj.dropna(how="any")
adj_prev = adj.shift(1).dropna(how="any")
common_idx = adj.index.intersection(adj_prev.index)  # 两者共有日期
adj = adj.loc[common_idx]

# ③ 计算对数收益率
logret = np.log(adj / adj.shift(1)).dropna()


# ---------- 3. 构造 GLD‑USO 等权组合 ----------
logret["GLD_USO"] = 0.5 * logret["GLD"] + 0.5 * logret["USO"]

# ---------- 4. 描述统计并保存 CSV ----------
rets = logret[["^GSPC", "GLD", "USO", "GLD_USO"]].copy()

desc = rets.describe().T
desc["skew"]     = rets.skew()
desc["kurtosis"] = rets.kurtosis()
print(desc)

desc.to_csv("shortdesc_stats.csv")

print("已保存 desc_stats.csv")

# ---------- 5. 绩效指标：Beta / Sharpe / Treynor / Jensen α ----------
# 5.1 日无风险收益率（^IRX 为年化百分比）
rf_daily = (adj["^IRX"] / 100 / 252).reindex(logret.index).ffill()


# 5.2 市场收益率（ACWI ETF）
rm = logret["ACWI"]

metrics = pd.DataFrame(index=["SP500", "GLD", "USO", "GLD_USO"],
                       columns=["Beta", "Sharpe", "Treynor", "Jensen_alpha"])

for name, col in zip(metrics.index, ["^GSPC", "GLD", "USO", "GLD_USO"]):
    r  = logret[col]
    ex = r - rf_daily                          # 超额收益

    beta   = r.cov(rm) / rm.var()
    sharpe = ex.mean() / ex.std() * np.sqrt(252)
    treynor = ex.mean() * 252 / beta
    alpha  = (r.mean()*252) - (rf_daily.mean()*252 + beta*(rm.mean()*252 - rf_daily.mean()*252))

    metrics.loc[name] = [beta, sharpe, treynor, alpha]

print("\n各资产（组合）绩效指标：")
print(metrics.round(4))
