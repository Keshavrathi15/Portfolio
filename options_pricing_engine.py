"""
Options Pricing Engine
======================
Prices European Call & Put options using:
  1. Black-Scholes model
  2. Binomial Tree model (Cox-Ross-Rubinstein)

Also computes all 5 Greeks:
  Delta, Gamma, Vega, Theta, Rho

Outputs a 4-panel chart:
  - Option price vs Spot price
  - All Greeks vs Spot price
  - Binomial vs Black-Scholes comparison
  - Implied Volatility surface (heatmap)

HOW TO RUN:
  python options_pricing_engine.py
  The chart saves as options_pricing_engine.png in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ── 0. Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FAFAFA",
    "axes.facecolor":    "#FAFAFA",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         10,
})

# ── 1. Base parameters ────────────────────────────────────────────────────────
# Think of these like inputs to a financial calculator.
# You can change any of these to price different options.

S = 19500      # Current spot price (e.g. NIFTY at 19,500)
K = 19500      # Strike price (at-the-money)
T = 30/365     # Time to expiry: 30 days expressed as fraction of a year
r = 0.065      # Risk-free rate: 6.5% (India 10Y gilt)
sigma = 0.18   # Implied volatility: 18% (typical for NIFTY options)

# ── 2. Black-Scholes Model ────────────────────────────────────────────────────
# The Black-Scholes formula prices options by assuming stock prices
# follow a random walk (log-normal distribution).
#
# d1 and d2 are intermediate values that feed into the normal distribution.
# N(d1) and N(d2) give the probability-weighted payoffs.

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Price a European option using Black-Scholes.

    Parameters:
        S    : Current stock/index price
        K    : Strike price
        T    : Time to expiry in years (e.g. 30/365 for 30 days)
        r    : Annual risk-free rate (e.g. 0.065 for 6.5%)
        sigma: Annual volatility (e.g. 0.18 for 18%)
        option_type: "call" or "put"

    Returns:
        price: Option price in same currency as S
    """
    if T <= 0:
        # At expiry: call = max(S-K, 0), put = max(K-S, 0)
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        # Put-Call Parity: Put = Call - S + K*e^(-rT)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# ── 3. Greeks ─────────────────────────────────────────────────────────────────
# Greeks measure how sensitive an option's price is to each input.
# Every options trader watches these daily.

def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all 5 Greeks for an option.

    Delta : Price change per ₹1 move in spot         (direction risk)
    Gamma : Rate of change of Delta                   (curvature risk)
    Vega  : Price change per 1% move in volatility    (vol risk)
    Theta : Price decay per day                       (time decay)
    Rho   : Price change per 1% move in interest rate (rate risk)
    """
    if T <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard normal PDF at d1 (bell curve height)
    nd1_pdf = norm.pdf(d1)

    # Delta: how much option price moves when spot moves ₹1
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1      # always negative for puts

    # Gamma: same formula for calls and puts
    gamma = nd1_pdf / (S * sigma * np.sqrt(T))

    # Vega: divide by 100 to express per 1% vol move
    vega = S * nd1_pdf * np.sqrt(T) / 100

    # Theta: daily time decay (divide by 365)
    if option_type == "call":
        theta = (-(S * nd1_pdf * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * nd1_pdf * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Rho: divide by 100 to express per 1% rate move
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma,
            "vega": vega,   "theta": theta, "rho": rho}


# ── 4. Binomial Tree (Cox-Ross-Rubinstein) ────────────────────────────────────
# An alternative to Black-Scholes that builds a tree of possible
# future prices step by step. More steps = more accurate.
# Converges to Black-Scholes as steps → infinity.

def binomial_tree(S, K, T, r, sigma, option_type="call", steps=200):
    """
    Price a European option using the CRR Binomial Tree.

    The tree works by:
    1. Splitting T into 'steps' equal time slices
    2. At each step, price goes UP by factor u or DOWN by factor d
    3. Working backwards from expiry to today using risk-neutral probabilities
    """
    dt = T / steps                         # length of each time step
    u  = np.exp(sigma * np.sqrt(dt))       # up factor
    d  = 1 / u                             # down factor (symmetric)
    p  = (np.exp(r * dt) - d) / (u - d)   # risk-neutral probability of up move

    # Build terminal price nodes (all possible prices at expiry)
    j       = np.arange(steps + 1)
    ST      = S * (u ** j) * (d ** (steps - j))

    # Compute payoff at expiry
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Discount backwards through the tree (vectorised — fast!)
    discount = np.exp(-r * dt)
    for _ in range(steps):
        payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1])

    return payoff[0]


# ── 5. Compute prices for the base case ──────────────────────────────────────
bs_call  = black_scholes(S, K, T, r, sigma, "call")
bs_put   = black_scholes(S, K, T, r, sigma, "put")
bt_call  = binomial_tree(S, K, T, r, sigma, "call")
bt_put   = binomial_tree(S, K, T, r, sigma, "put")
greeks_c = compute_greeks(S, K, T, r, sigma, "call")
greeks_p = compute_greeks(S, K, T, r, sigma, "put")

print("\n" + "="*55)
print("  OPTIONS PRICING ENGINE — BASE CASE")
print(f"  Spot={S}  Strike={K}  T=30d  r=6.5%  σ=18%")
print("="*55)
print(f"\n  {'':12}  {'CALL':>10}  {'PUT':>10}")
print(f"  {'-'*36}")
print(f"  {'Black-Scholes':12}  {bs_call:>10.2f}  {bs_put:>10.2f}")
print(f"  {'Binomial Tree':12}  {bt_call:>10.2f}  {bt_put:>10.2f}")
print(f"  {'Difference':12}  {abs(bs_call-bt_call):>10.4f}  {abs(bs_put-bt_put):>10.4f}")

print("\n  GREEKS (Call / Put)")
print(f"  {'-'*36}")
for g in ["delta", "gamma", "vega", "theta", "rho"]:
    print(f"  {g.capitalize():<8}  {greeks_c[g]:>10.4f}  {greeks_p[g]:>10.4f}")


# ── 6. Build data for charts ──────────────────────────────────────────────────
# Vary spot price from 80% to 120% of current price
spot_range = np.linspace(S * 0.80, S * 1.20, 200)

# Option prices across spot range
call_prices = [black_scholes(s, K, T, r, sigma, "call") for s in spot_range]
put_prices  = [black_scholes(s, K, T, r, sigma, "put")  for s in spot_range]

# Greeks across spot range
greek_names  = ["delta", "gamma", "vega", "theta", "rho"]
greek_colors = ["#1D9E75", "#534AB7", "#D85A30", "#D4537E", "#888780"]
greeks_by_spot = {g: [] for g in greek_names}
for s in spot_range:
    g = compute_greeks(s, K, T, r, sigma, "call")
    for name in greek_names:
        greeks_by_spot[name].append(g[name])

# Binomial vs BS across step counts
step_range  = range(5, 205, 5)
bt_call_steps = [binomial_tree(S, K, T, r, sigma, "call", steps=n) for n in step_range]
bt_put_steps  = [binomial_tree(S, K, T, r, sigma, "put",  steps=n) for n in step_range]

# Implied Volatility surface: price grid over strikes x maturities
strikes    = np.linspace(S * 0.85, S * 1.15, 12)
maturities = np.array([7, 14, 21, 30, 45, 60, 90]) / 365
iv_surface = np.zeros((len(maturities), len(strikes)))

# Simulate a simple volatility smile: higher vol at extremes (realistic)
for i, t in enumerate(maturities):
    for j, k in enumerate(strikes):
        moneyness    = np.log(k / S)
        smile_adj    = 0.04 * moneyness**2    # parabolic smile
        term_adj     = 0.02 * np.sqrt(t)      # term structure
        iv_surface[i, j] = (sigma + smile_adj - term_adj) * 100


# ── 7. 4-Panel Chart ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Options Pricing Engine — NIFTY European Options",
             fontsize=16, fontweight="bold", y=0.98, color="#1A1A1A")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

C = {"call": "#1D9E75", "put": "#534AB7", "bs": "#1D9E75",
     "bt": "#D85A30",   "atm": "#AAAAAA"}

# ── Panel 1: Call & Put price vs Spot ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(spot_range, call_prices, color=C["call"], lw=2.0, label="Call price")
ax1.plot(spot_range, put_prices,  color=C["put"],  lw=2.0, label="Put price")
ax1.axvline(K, color=C["atm"], lw=1.0, linestyle="--", label=f"Strike = {K:,}")
ax1.axvline(S, color="#CCCCCC", lw=0.8, linestyle=":")
ax1.scatter([S], [bs_call], color=C["call"], s=80, zorder=5)
ax1.scatter([S], [bs_put],  color=C["put"],  s=80, zorder=5)
ax1.annotate(f"Call ₹{bs_call:.0f}", xy=(S, bs_call),
             xytext=(10, 5), textcoords="offset points", fontsize=9, color=C["call"])
ax1.annotate(f"Put ₹{bs_put:.0f}",  xy=(S, bs_put),
             xytext=(10, -14), textcoords="offset points", fontsize=9, color=C["put"])
ax1.set_xlabel("Spot Price (₹)")
ax1.set_ylabel("Option Price (₹)")
ax1.set_title("Black-Scholes: Call & Put price vs Spot", fontsize=11)
ax1.legend(fontsize=9)

# ── Panel 2: Greeks vs Spot ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for name, color in zip(greek_names, greek_colors):
    ax2.plot(spot_range, greeks_by_spot[name], color=color, lw=1.6, label=name.capitalize())
ax2.axvline(K, color=C["atm"], lw=1.0, linestyle="--")
ax2.axhline(0, color="#DDDDDD", lw=0.8)
ax2.set_xlabel("Spot Price (₹)")
ax2.set_ylabel("Greek value")
ax2.set_title("Call Greeks vs Spot Price", fontsize=11)
ax2.legend(fontsize=9, ncol=2)

# ── Panel 3: Binomial Tree convergence to Black-Scholes ──────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(list(step_range), bt_call_steps, color=C["bt"],  lw=1.8, label="Binomial call")
ax3.plot(list(step_range), bt_put_steps,  color=C["put"], lw=1.8, label="Binomial put", linestyle="--")
ax3.axhline(bs_call, color=C["call"],  lw=1.2, linestyle=":", label=f"BS call = ₹{bs_call:.2f}")
ax3.axhline(bs_put,  color="#3C3489",  lw=1.2, linestyle=":", label=f"BS put  = ₹{bs_put:.2f}")
ax3.set_xlabel("Number of tree steps")
ax3.set_ylabel("Option Price (₹)")
ax3.set_title("Binomial Tree convergence to Black-Scholes", fontsize=11)
ax3.legend(fontsize=9)

# ── Panel 4: Implied Volatility Surface ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
mat_labels    = ["7d", "14d", "21d", "30d", "45d", "60d", "90d"]
strike_labels = [f"{int(k/1000)}K" for k in strikes]
im = ax4.imshow(iv_surface, cmap="YlOrRd", aspect="auto", origin="lower")
plt.colorbar(im, ax=ax4, label="Implied Volatility (%)", shrink=0.85)
ax4.set_xticks(range(len(strikes)))
ax4.set_xticklabels(strike_labels, fontsize=8, rotation=45)
ax4.set_yticks(range(len(maturities)))
ax4.set_yticklabels(mat_labels, fontsize=8)
ax4.set_xlabel("Strike Price")
ax4.set_ylabel("Time to Expiry")
ax4.set_title("Implied Volatility Surface (Vol Smile)", fontsize=11)

# Annotate a few IV values
for i in range(len(maturities)):
    for j in [0, 5, 11]:
        ax4.text(j, i, f"{iv_surface[i,j]:.1f}",
                 ha="center", va="center", fontsize=7.5, color="#333333")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig("options_pricing_engine.png", dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print("\nChart saved → options_pricing_engine.png")
plt.close()
