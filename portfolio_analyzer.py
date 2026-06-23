import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import minimize

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.size":        10,
})

np.random.seed(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="B")
n = len(dates)

STOCK_PARAMS = {
    "RELIANCE":  (0.00055, 0.016, 2400),
    "TCS":       (0.00045, 0.014, 3200),
    "INFY":      (0.00038, 0.015, 1450),
    "HDFCBANK":  (0.00042, 0.013, 1550),
    "WIPRO":     (0.00030, 0.017,  420),
    "BAJFINANCE":(0.00060, 0.020, 6500),
    "ICICIBANK": (0.00050, 0.015,  900),
    "MARUTI":    (0.00048, 0.016, 9800),
    "SUNPHARMA": (0.00040, 0.014,  980),
    "ITC":       (0.00035, 0.012,  340),
}

NIFTY_PARAMS = (0.00040, 0.013, 17000)

raw = {}
for name, (mu, sigma, start) in STOCK_PARAMS.items():
    ret = np.random.normal(mu, sigma, n)
    raw[name] = start * np.cumprod(1 + ret)

nifty_ret = np.random.normal(*NIFTY_PARAMS[:2], n)
nifty_prices = NIFTY_PARAMS[2] * np.cumprod(1 + nifty_ret)

prices  = pd.DataFrame(raw, index=dates)
nifty   = pd.Series(nifty_prices, index=dates, name="NIFTY50")

daily_ret   = prices.pct_change().dropna()
nifty_daily = nifty.pct_change().dropna()

TRADING_DAYS = 252
RF_DAILY     = 0.065 / TRADING_DAYS       

ann_return  = daily_ret.mean()  * TRADING_DAYS
ann_vol     = daily_ret.std()   * np.sqrt(TRADING_DAYS)
sharpe      = (daily_ret.mean() - RF_DAILY) / daily_ret.std() * np.sqrt(TRADING_DAYS)

market_var  = nifty_daily.var()
beta        = daily_ret.apply(lambda col: col.cov(nifty_daily) / market_var)

def max_drawdown(series):
    cum   = (1 + series).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    return dd.min()

mdd = daily_ret.apply(max_drawdown)

metrics = pd.DataFrame({
    "Ann. Return (%)":    (ann_return * 100).round(2),
    "Ann. Volatility (%)": (ann_vol   * 100).round(2),
    "Sharpe Ratio":        sharpe.round(2),
    "Beta (vs Nifty)":     beta.round(2),
    "Max Drawdown (%)":   (mdd * 100).round(2),
})

print("\n" + "="*65)
print("  INDIVIDUAL STOCK METRICS")
print("="*65)
print(metrics.to_string())

N_STOCKS  = len(prices.columns)
N_SIM     = 10_000
cov_matrix = daily_ret.cov() * TRADING_DAYS
mean_ret   = daily_ret.mean() * TRADING_DAYS

sim_returns, sim_vols, sim_sharpes, sim_weights = [], [], [], []

for _ in range(N_SIM):
    w = np.random.dirichlet(np.ones(N_STOCKS))
    r = np.dot(w, mean_ret)
    v = np.sqrt(w @ cov_matrix.values @ w)
    s = (r - 0.065) / v
    sim_returns.append(r)
    sim_vols.append(v)
    sim_sharpes.append(s)
    sim_weights.append(w)

sim_returns  = np.array(sim_returns)
sim_vols     = np.array(sim_vols)
sim_sharpes  = np.array(sim_sharpes)
sim_weights  = np.array(sim_weights)

max_sharpe_idx = np.argmax(sim_sharpes)
min_vol_idx    = np.argmin(sim_vols)

opt_weights_sharpe = sim_weights[max_sharpe_idx]
opt_weights_minvol = sim_weights[min_vol_idx]

print("\n" + "="*65)
print("  OPTIMAL PORTFOLIO — MAX SHARPE RATIO")
print("="*65)
for stock, w in zip(prices.columns, opt_weights_sharpe):
    print(f"  {stock:<12} {w*100:6.2f}%")
print(f"\n  Expected Return : {sim_returns[max_sharpe_idx]*100:.2f}%")
print(f"  Volatility      : {sim_vols[max_sharpe_idx]*100:.2f}%")
print(f"  Sharpe Ratio    : {sim_sharpes[max_sharpe_idx]:.2f}")

print("\n" + "="*65)
print("  OPTIMAL PORTFOLIO — MINIMUM VOLATILITY")
print("="*65)
for stock, w in zip(prices.columns, opt_weights_minvol):
    print(f"  {stock:<12} {w*100:6.2f}%")
print(f"\n  Expected Return : {sim_returns[min_vol_idx]*100:.2f}%")
print(f"  Volatility      : {sim_vols[min_vol_idx]*100:.2f}%")
print(f"  Sharpe Ratio    : {sim_sharpes[min_vol_idx]:.2f}")

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Portfolio Risk & Return Analyzer — NSE Equities",
             fontsize=16, fontweight="bold", y=0.98, color="#1A1A1A")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

COLORS = {
    "scatter": "#B5D4F4",  
    "max_sr":  "#1D9E75",   
    "min_vol": "#534AB7",   
    "nifty":   "#D85A30",   
}

ax1 = fig.add_subplot(gs[0, 0])
sc  = ax1.scatter(sim_vols * 100, sim_returns * 100,
                  c=sim_sharpes, cmap="YlGn", alpha=0.35, s=6, linewidths=0)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio", shrink=0.85)

ax1.scatter(sim_vols[max_sharpe_idx] * 100, sim_returns[max_sharpe_idx] * 100,
            color=COLORS["max_sr"], s=140, zorder=5, label="Max Sharpe", marker="*")
ax1.scatter(sim_vols[min_vol_idx] * 100, sim_returns[min_vol_idx] * 100,
            color=COLORS["min_vol"], s=100, zorder=5, label="Min Volatility", marker="D")

ax1.set_xlabel("Annualised Volatility (%)")
ax1.set_ylabel("Annualised Return (%)")
ax1.set_title("Efficient Frontier (10,000 simulations)", fontsize=11)
ax1.legend(fontsize=9)

ax2  = fig.add_subplot(gs[0, 1])
corr = daily_ret.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax2, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.4, linecolor="#E0E0E0",
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax2.set_title("Return Correlation Matrix", fontsize=11)
ax2.tick_params(axis="x", rotation=45, labelsize=8)
ax2.tick_params(axis="y", rotation=0,  labelsize=8)

ax3     = fig.add_subplot(gs[1, 0])
labels  = list(prices.columns)
weights = opt_weights_sharpe * 100
sorted_idx = np.argsort(weights)[::-1]
bar_colors = [COLORS["max_sr"] if w > 10 else "#9FE1CB" for w in weights[sorted_idx]]

bars = ax3.barh([labels[i] for i in sorted_idx], weights[sorted_idx],
                color=bar_colors, edgecolor="none", height=0.65)
for bar, val in zip(bars, weights[sorted_idx]):
    ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9, color="#1A1A1A")
ax3.set_xlabel("Portfolio Weight (%)")
ax3.set_title(f"Max Sharpe Portfolio  (SR = {sim_sharpes[max_sharpe_idx]:.2f})", fontsize=11)
ax3.set_xlim(0, max(weights) * 1.2)

ax4 = fig.add_subplot(gs[1, 1])

ew_weights   = np.ones(N_STOCKS) / N_STOCKS
ew_daily_ret = (daily_ret * ew_weights).sum(axis=1)
ew_cum       = (1 + ew_daily_ret).cumprod()

ms_daily_ret = (daily_ret * opt_weights_sharpe).sum(axis=1)
ms_cum       = (1 + ms_daily_ret).cumprod()

nifty_cum = (1 + nifty_daily).cumprod()

ax4.plot(ms_cum.index,  ms_cum.values,    color=COLORS["max_sr"],  lw=2.0, label="Max Sharpe Portfolio")
ax4.plot(ew_cum.index,  ew_cum.values,    color=COLORS["min_vol"], lw=1.5, label="Equal Weight Portfolio", linestyle="--")
ax4.plot(nifty_cum.index, nifty_cum.values, color=COLORS["nifty"], lw=1.5, label="Nifty 50", linestyle=":")
ax4.axhline(1, color="#AAAAAA", linewidth=0.8, linestyle="-")
ax4.set_ylabel("Cumulative Return (₹1 invested)")
ax4.set_title("Portfolio Performance vs Nifty 50  (3Y)", fontsize=11)
ax4.legend(fontsize=9)
ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

for cum, label, color in [
    (ms_cum, "Max Sharpe", COLORS["max_sr"]),
    (ew_cum, "Equal Wt",   COLORS["min_vol"]),
    (nifty_cum, "Nifty",   COLORS["nifty"]),
]:
    final = cum.iloc[-1]
    ax4.annotate(f"{label}: {(final-1)*100:.0f}%",
                 xy=(cum.index[-1], final),
                 xytext=(8, 0), textcoords="offset points",
                 fontsize=8, color=color, va="center")

out_path = "/mnt/user-data/outputs/portfolio_analyzer.png"
plt.savefig("portfolio_analyzer.png", dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print(f"\nChart saved → {out_path}")
plt.close()