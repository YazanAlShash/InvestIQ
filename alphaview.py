"""
AlphaView — Investment Analysis Platform
DAB401 Final Group Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
from scipy import stats
from datetime import datetime, timedelta
import gc
import warnings
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="AlphaView — Investment Analysis", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
[data-testid="stSidebar"] { background: #080c14 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
[data-testid="stSidebar"] .stButton button { background: #3b82f6 !important; border: none !important; border-radius: 8px !important; color: #fff !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stButton button:hover { background: #2563eb !important; }
.main, .stApp { background: #080c14 !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; gap: 2px !important; border: 1px solid rgba(255,255,255,0.06) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 7px !important; color: rgba(255,255,255,0.4) !important; font-weight: 500 !important; font-size: 13px !important; padding: 6px 16px !important; border: none !important; }
.stTabs [aria-selected="true"] { background: rgba(255,255,255,0.08) !important; color: #fff !important; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; padding: 14px 16px !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.4) !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'DM Mono', monospace !important; font-size: 22px !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; }
.stApp, .main .block-container { background: #080c14 !important; }
h1, h2, h3 { color: #fff !important; letter-spacing: -0.3px !important; }
p, li { color: rgba(255,255,255,0.75) !important; }
.stCaption { color: rgba(255,255,255,0.3) !important; font-size: 11px !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly shared config ──────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
    font=dict(family="DM Sans, sans-serif", color="rgba(255,255,255,0.7)", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="rgba(255,255,255,0.6)")),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1d27", bordercolor="rgba(255,255,255,0.1)",
                    font=dict(family="DM Mono, monospace", size=12)),
)

RS = dict(
    buttons=[
        dict(count=1,  label="1M", step="month", stepmode="backward"),
        dict(count=3,  label="3M", step="month", stepmode="backward"),
        dict(count=6,  label="6M", step="month", stepmode="backward"),
        dict(count=1,  label="1Y", step="year",  stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor="rgba(255,255,255,0.04)", activecolor="rgba(59,130,246,0.5)",
    bordercolor="rgba(255,255,255,0.08)", font=dict(color="rgba(255,255,255,0.6)", size=11),
)
RL = dict(visible=True, bgcolor="#080c14", bordercolor="rgba(255,255,255,0.06)", thickness=0.04)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px 0">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <svg width="32" height="32" viewBox="0 0 92 104">
                <polygon points="46,0 92,26 92,78 46,104 0,78 0,26" fill="#0f1f3d" stroke="#1e3a6e" stroke-width="1.5"/>
                <rect x="14" y="62" width="9" height="20" rx="2" fill="#1d4ed8"/>
                <rect x="27" y="50" width="9" height="32" rx="2" fill="#2563eb"/>
                <rect x="40" y="38" width="9" height="44" rx="2" fill="#3b82f6"/>
                <rect x="53" y="28" width="9" height="54" rx="2" fill="#60a5fa"/>
                <polyline points="14,72 27,58 40,44 62,24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="62" cy="24" r="4" fill="#22c55e"/>
            </svg>
            <div>
                <div style="font-size:16px;font-weight:600;color:#fff;letter-spacing:-0.3px;line-height:1">
                    Alpha<span style="color:#3b82f6;font-weight:300">View</span>
                </div>
                <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px">
                    Investment Analysis
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🎯 Stock**")
    ticker_input = st.text_input("Ticker", value="", placeholder="e.g. AAPL, AMZN, BTC-CAD",
                                  label_visibility="collapsed").upper().strip()
    st.markdown("**📅 Historical Data**")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=None)
    with c2: end_date   = st.date_input("To",   value=datetime.today())

    st.markdown("**🎲 Monte Carlo**")
    prediction_days = st.number_input("Forecast (days)", 30, 2000, 365, 10)
    with st.expander("⚙️ Simulation settings"):
        iter_map = {"1,000 — Fast":1_000,"5,000":5_000,"10,000 — Default":10_000,
                    "50,000":50_000,"100,000 — Slow":100_000}
        iterations = iter_map[st.selectbox("Iterations", list(iter_map.keys()), index=2)]

    st.markdown("**👤 Your Profile**")
    risk_tolerance = st.selectbox("Risk Tolerance", ["Low","Medium","High"])
    horizon        = st.selectbox("Investment Horizon",
                                  ["Short-term (< 1 yr)","Medium-term (1–3 yrs)","Long-term (3+ yrs)"])
    invest_amount  = st.number_input("Investment ($)", 100, 10_000_000, 10_000, 500)
    with st.expander("📐 CAPM Parameters"):
        rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100

    st.divider()
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")

# ── Helpers ───────────────────────────────────────────────────
def sg(d,k,default=None):
    try: v=d.get(k,default); return v if v is not None else default
    except: return default

def sv(df,row,col=0):
    try:
        if df is None or df.empty or row not in df.index: return None
        v=df.loc[row].iloc[col]; return float(v) if pd.notna(v) else None
    except: return None

@st.cache_data(ttl=3600)
def fetch_all(ticker, start, end):
    t=yf.Ticker(ticker); hist=t.history(start=start,end=end,auto_adjust=True)
    if hist.empty: return None
    info=t.info; income=t.financials; balance=t.balance_sheet; cashflow=t.cashflow
    try: rf=yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]/100
    except: rf=0.045
    try:
        sp=yf.Ticker("^GSPC").history(period="10y")["Close"]
        rm=float(sp.resample("YE").last().pct_change().dropna().mean())
    except: rm=0.10
    return dict(hist=hist,info=info,income=income,balance=balance,cashflow=cashflow,rf=rf,rm=rm)

def calc_ratios(info,income,balance,cashflow):
    r={}
    rev=sv(income,"Total Revenue"); gross=sv(income,"Gross Profit")
    op=sv(income,"Operating Income"); net=sv(income,"Net Income")
    ca=sv(balance,"Current Assets"); cl=sv(balance,"Current Liabilities")
    inv=sv(balance,"Inventory") or 0
    cash=sv(balance,"Cash And Cash Equivalents") or sv(balance,"Cash Cash Equivalents And Short Term Investments") or 0
    ta=sv(balance,"Total Assets")
    eq=sv(balance,"Stockholders Equity") or sv(balance,"Common Stock Equity")
    dbt=sv(balance,"Total Debt") or sv(balance,"Long Term Debt") or 0
    ocf=sv(cashflow,"Operating Cash Flow") or sv(cashflow,"Cash Flow From Continuing Operating Activities")
    capex=sv(cashflow,"Capital Expenditure") or 0
    if capex>0: capex=-capex
    def p(a,b): return a/b*100 if a and b else None
    def d(a,b): return a/b if a and b else None
    r["Gross Profit Margin"]=p(gross,rev); r["Operating Profit Margin"]=p(op,rev)
    r["Net Profit Margin"]=p(net,rev); r["Return on Assets (ROA)"]=p(net,ta)
    r["Return on Equity (ROE)"]=p(net,eq); r["EPS"]=sg(info,"trailingEps")
    r["Current Ratio"]=d(ca,cl); r["Quick Ratio"]=d(ca-inv,cl) if ca and cl else None
    r["Cash Ratio"]=d(cash,cl); r["P/E Ratio"]=sg(info,"trailingPE")
    r["Dividend Payout Ratio"]=sg(info,"payoutRatio"); r["Debt-to-Equity"]=d(dbt,eq)
    roe=r.get("Return on Equity (ROE)"); pr=r.get("Dividend Payout Ratio")
    r["Sustainable Growth Rate"]=roe/100*(1-pr)*100 if roe and pr is not None else None
    r["Free Cash Flow"]=ocf+capex if ocf else None
    return r

def calc_beta(hist):
    try:
        sp=yf.Ticker("^GSPC").history(start=(datetime.today()-timedelta(days=365*5)).strftime("%Y-%m-%d"),
                                      end=datetime.today().strftime("%Y-%m-%d"),auto_adjust=True)["Close"]
        sw=hist["Close"].resample("W").last(); mw=sp.resample("W").last()
        df=pd.DataFrame({"s":sw,"m":mw}).dropna()
        sr=df["s"].pct_change().dropna(); mr=df["m"].pct_change().dropna()
        df2=pd.DataFrame({"s":sr,"m":mr}).dropna()
        b,ic,rv,_,_=stats.linregress(df2["m"],df2["s"])
        return b,rv**2,df2
    except: return None,None,None

def calc_wacc(info,balance,rf,rm,beta):
    try:
        mc=sg(info,"marketCap"); tax=sg(info,"effectiveTaxRate") or 0.21
        dbt=sv(balance,"Total Debt") or sv(balance,"Long Term Debt") or 0
        if not mc or not beta: return None,None,None
        re=rf+beta*(rm-rf); rd=abs(sg(info,"interestExpense") or 0)/dbt if dbt>0 else 0.05
        V=mc+dbt; wacc=(mc/V)*re+(dbt/V)*rd*(1-tax)
        return wacc,re,rd
    except: return None,None,None

def calc_intrinsic(cashflow,info,wacc,g=0.03):
    try:
        fcf=sv(cashflow,"Free Cash Flow") or sv(cashflow,"Operating Cash Flow")
        sh=sg(info,"sharesOutstanding")
        if not fcf or not wacc or wacc<=g or not sh: return None
        return (fcf*(1+g)/(wacc-g))/sh
    except: return None

def run_mc(close_series,days,n_iter,chunk=5000):
    s=close_series.squeeze().dropna(); rets=np.log(1+s.pct_change()).dropna()
    u=float(rets.mean()); var=float(rets.var())
    drift=u-0.5*var; stdev=float(rets.std()); S0=float(s.iloc[-1])
    finals=[]; ns=min(200,n_iter); sp=np.zeros((days,ns)); filled=0; rem=n_iter
    while rem>0:
        b=min(chunk,rem); rem-=b
        Z=norm.ppf(np.random.rand(days,b)); dr=np.exp(drift+stdev*Z)
        p=np.zeros((days,b)); p[0]=S0
        for t in range(1,days): p[t]=p[t-1]*dr[t]
        finals.append(p[-1])
        take=min(b,ns-filled)
        if take>0: sp[:,filled:filled+take]=p[:,:take]; filled+=take
        del Z,dr,p; gc.collect()
    return S0,np.concatenate(finals),sp

def build_scenario(S0,p5,p25,p50,p75,p95,ratios,capm_ret,cur,intrinsic,risk,hor,amt,days):
    sigs=[]
    mc_p=(p50-S0)/S0*100
    sigs.append(("MC","positive" if mc_p>5 else ("negative" if mc_p<-5 else "neutral"),f"MC median {mc_p:+.1f}%"))
    if capm_ret: sigs.append(("CAPM","positive" if capm_ret>0.08 else ("negative" if capm_ret<0.03 else "neutral"),f"CAPM {capm_ret*100:.1f}%"))
    if intrinsic and cur:
        gap=(intrinsic-cur)/cur*100
        sigs.append(("VAL","positive" if gap>15 else ("negative" if gap<-15 else "neutral"),f"Intrinsic ${intrinsic:.2f} ({gap:+.1f}%)"))
    roe=ratios.get("Return on Equity (ROE)"); cr=ratios.get("Current Ratio"); de=ratios.get("Debt-to-Equity")
    rs=sum([1 if roe and roe>15 else(-1 if roe and roe<0 else 0),
            1 if cr and cr>1.5 else(-1 if cr and cr<1 else 0),
            1 if de and de<1 else(-1 if de and de>3 else 0)])
    sigs.append(("RATIO","positive" if rs>=2 else("negative" if rs<=-1 else "neutral"),
                 "Strong fundamentals" if rs>=2 else("Weak ratios" if rs<=-1 else "Mixed ratios")))
    pos=sum(1 for _,s,_ in sigs if s=="positive"); neg=sum(1 for _,s,_ in sigs if s=="negative")
    rm_={"Low":0.7,"Medium":1.0,"High":1.3}[risk]
    hm_={"Short-term (< 1 yr)":0.8,"Medium-term (1–3 yrs)":1.0,"Long-term (3+ yrs)":1.2}[hor]
    bp=p75*rm_*hm_; base=p50; bear=p25/rm_
    return dict(signals=sigs,pos=pos,neg=neg,
                bull_price=bp,base_price=base,bear_price=bear,
                bull_ret=(bp-S0)/S0,base_ret=(base-S0)/S0,bear_ret=(bear-S0)/S0,
                bull_gain=amt*(bp-S0)/S0,base_gain=amt*(base-S0)/S0,bear_gain=amt*(bear-S0)/S0,
                risk_level="🔴 High" if abs((p5-S0)/S0)>0.4 else("🟡 Medium" if abs((p5-S0)/S0)>0.2 else "🟢 Low"),
                downside_pct=abs((p5-S0)/S0*100))

def _sma(a,n): return pd.Series(a).rolling(n).mean().values
def _ema(a,n): return pd.Series(a).ewm(span=n,adjust=False).mean().values
def _wma(a,n):
    w=np.arange(1,n+1)
    return pd.Series(a).rolling(n).apply(lambda x:np.dot(x,w)/w.sum(),raw=True).values
def _rsi(a,p=14):
    d=pd.Series(a).diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean(); return (100-100/(1+g/l)).values
def _crosses(fast,slow):
    bull,bear=[],[]
    for i in range(1,len(fast)):
        if np.isnan(fast[i]) or np.isnan(slow[i]): continue
        if fast[i]>slow[i] and fast[i-1]<=slow[i-1]: bull.append(i)
        elif fast[i]<slow[i] and fast[i-1]>=slow[i-1]: bear.append(i)
    return bull,bear

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
    <svg width="36" height="36" viewBox="0 0 92 104">
        <polygon points="46,0 92,26 92,78 46,104 0,78 0,26" fill="#0f1f3d" stroke="#1e3a6e" stroke-width="1.5"/>
        <rect x="14" y="62" width="9" height="20" rx="2" fill="#1d4ed8"/>
        <rect x="27" y="50" width="9" height="32" rx="2" fill="#2563eb"/>
        <rect x="40" y="38" width="9" height="44" rx="2" fill="#3b82f6"/>
        <rect x="53" y="28" width="9" height="54" rx="2" fill="#60a5fa"/>
        <polyline points="14,72 27,58 40,44 62,24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="62" cy="24" r="4" fill="#22c55e"/>
    </svg>
    <div>
        <h1 style="margin:0;font-size:24px;font-weight:600;letter-spacing:-0.5px;line-height:1">
            Alpha<span style="color:#3b82f6;font-weight:300">View</span>
        </h1>
        <span style="color:rgba(255,255,255,0.25);font-size:12px;letter-spacing:1.5px;text-transform:uppercase">
            Investment Analysis Platform
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div style="background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.2);
                border-radius:10px;padding:20px 24px;margin-top:12px">
        <p style="color:rgba(255,255,255,0.6);margin:0;font-size:14px">
            👈 Enter a ticker and your profile in the sidebar, then click
            <strong style="color:#3b82f6">Run Full Analysis</strong>.
        </p>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px">
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:16px">
            <div style="font-size:18px;margin-bottom:8px">📊</div>
            <div style="font-weight:500;color:#fff;font-size:13px;margin-bottom:4px">Financial Ratios</div>
            <div style="color:rgba(255,255,255,0.35);font-size:12px">Profitability, liquidity, P/E, FCF and more</div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:16px">
            <div style="font-size:18px;margin-bottom:8px">🎲</div>
            <div style="font-weight:500;color:#fff;font-size:13px;margin-bottom:4px">Monte Carlo</div>
            <div style="color:rgba(255,255,255,0.35);font-size:12px">Simulated price paths with confidence bands</div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:16px">
            <div style="font-size:18px;margin-bottom:8px">📉</div>
            <div style="font-weight:500;color:#fff;font-size:13px;margin-bottom:4px">Technical Analysis</div>
            <div style="color:rgba(255,255,255,0.35);font-size:12px">SMA, EMA, WMA crossovers + RSI + MACD</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not ticker_input:
    st.error("❌ Please enter a ticker symbol."); st.stop()

eff_start = start_date if start_date else datetime(1980, 1, 1).date()

with st.spinner(f"Fetching data for **{ticker_input}**..."):
    data = fetch_all(ticker_input, eff_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if data is None:
    st.error(f"❌ No data found for **{ticker_input}**."); st.stop()

hist=data["hist"]; info=data["info"]; income=data["income"]
balance=data["balance"]; cashflow=data["cashflow"]
rf_live=data["rf"]; rm=data["rm"]
rf=rf_rate if abs(rf_rate-0.045)>0.001 else rf_live
cur=float(hist["Close"].iloc[-1])
company=sg(info,"longName") or ticker_input

with st.spinner("Running calculations..."):
    ratios=calc_ratios(info,income,balance,cashflow)
    beta,r2,df_reg=calc_beta(hist)
    wacc,re,rd=calc_wacc(info,balance,rf,rm,beta)
    intrinsic=calc_intrinsic(cashflow,info,wacc)

with st.spinner(f"Running {iterations:,} Monte Carlo simulations..."):
    S0,finals,spaths=run_mc(hist["Close"],prediction_days,iterations)

p5=np.percentile(finals,5); p25=np.percentile(finals,25); p50=np.percentile(finals,50)
p75=np.percentile(finals,75); p95=np.percentile(finals,95)
p5p=np.percentile(spaths,5,axis=1); p25p=np.percentile(spaths,25,axis=1)
p50p=np.percentile(spaths,50,axis=1); p75p=np.percentile(spaths,75,axis=1)
p95p=np.percentile(spaths,95,axis=1)
tgt=(datetime.today()+timedelta(days=prediction_days)).strftime("%B %d, %Y")
scen=build_scenario(S0,p5,p25,p50,p75,p95,ratios,re,cur,intrinsic,
                    risk_tolerance,horizon,invest_amount,prediction_days)

_ma_base={"Short-term (< 1 yr)":(10,20),"Medium-term (1–3 yrs)":(20,50),"Long-term (3+ yrs)":(50,200)}
mf,ms=_ma_base[horizon]
if risk_tolerance=="Low":   mf=int(mf*1.2); ms=int(ms*1.2)
elif risk_tolerance=="High": mf=max(5,int(mf*0.8)); ms=max(10,int(ms*0.8))

# ── Tabs ──────────────────────────────────────────────────────
t0,t1,t2,t3,t4,t5,t6=st.tabs([
    "🏠 Overview","📊 Ratios","📐 CAPM & Valuation",
    "🎲 Monte Carlo","🔮 Prophet","📉 Technical","ℹ️ About"])

# ══════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════
with t0:
    mc_pct=(p50-S0)/S0*100; mc_c="#22c55e" if mc_pct>=0 else "#ef4444"
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-size:11px;color:rgba(255,255,255,0.3);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px">
            {ticker_input} · {sg(info,'exchange','')}
        </div>
        <div style="font-size:26px;font-weight:600;color:#fff;letter-spacing:-0.5px;margin-bottom:6px">{company}</div>
        <div style="display:flex;align-items:baseline;gap:10px">
            <span style="font-family:'DM Mono',monospace;font-size:30px;font-weight:500;color:#fff">${cur:.2f}</span>
            <span style="font-size:13px;font-weight:500;color:{mc_c};background:{'rgba(34,197,94,0.1)' if mc_pct>=0 else 'rgba(239,68,68,0.1)'};padding:3px 8px;border-radius:5px">MC {mc_pct:+.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols_clr={"positive":("#22c55e","rgba(34,197,94,0.1)","rgba(34,197,94,0.25)"),
              "negative":("#ef4444","rgba(239,68,68,0.1)","rgba(239,68,68,0.25)"),
              "neutral": ("#f59e0b","rgba(245,158,11,0.1)","rgba(245,158,11,0.25)")}
    pills=""
    for nm,sent,desc in scen["signals"]:
        c,bg,bc=cols_clr[sent]
        pills+=f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;padding:4px 10px;border-radius:20px;border:1px solid {bc};background:{bg};color:{c};margin:0 4px 4px 0"><span style="width:5px;height:5px;border-radius:50%;background:{c};display:inline-block"></span>{nm}: {desc}</span>'
    st.markdown(f'<div style="margin-bottom:20px">{pills}</div>',unsafe_allow_html=True)

    mc1,mc2,mc3,mc4,mc5,mc6=st.columns(6)
    mc1.metric("Current Price",f"${cur:.2f}")
    mc2.metric("MC Median",f"${p50:.2f}",f"{mc_pct:+.1f}%")
    mc3.metric("Beta",f"{beta:.2f}" if beta else "N/A")
    mc4.metric("CAPM Return",f"{re*100:.1f}%" if re else "N/A")
    mc5.metric("Intrinsic Value",f"${intrinsic:.2f}" if intrinsic else "N/A")
    mc6.metric("Risk-Free Rate",f"{rf*100:.2f}%")

    st.divider()
    st.markdown("### Scenario Analysis")
    sc1,sc2,sc3=st.columns(3)
    for col,lbl,price,ret,gain,clr,brd in [
        (sc1,"🟢 Bullish",scen["bull_price"],scen["bull_ret"],scen["bull_gain"],"#22c55e","rgba(34,197,94,0.2)"),
        (sc2,"🔵 Base Case",scen["base_price"],scen["base_ret"],scen["base_gain"],"#3b82f6","rgba(59,130,246,0.2)"),
        (sc3,"🔴 Bearish",scen["bear_price"],scen["bear_ret"],scen["bear_gain"],"#ef4444","rgba(239,68,68,0.15)"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid {brd};border-radius:10px;padding:16px">
            <div style="font-size:12px;font-weight:600;color:{clr};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">{lbl}</div>
            <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:500;color:#fff;margin-bottom:4px">${price:.2f}</div>
            <div style="font-size:13px;color:{clr};margin-bottom:6px">{ret*100:+.1f}% return</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.35)">Est. {'+' if gain>=0 else ''}${gain:,.0f} on ${invest_amount:,}</div>
        </div>""",unsafe_allow_html=True)

    st.divider()
    st.markdown("### Historical Price")
    fig_h=go.Figure()
    fig_h.add_trace(go.Scatter(x=hist.index,y=hist["Close"].values,mode="lines",name="Close",
        line=dict(color="#3b82f6",width=1.5),fill="tozeroy",fillcolor="rgba(59,130,246,0.05)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:.2f}<extra></extra>"))
    fig_h.update_layout(**PL,height=280,
        title=dict(text=f"{ticker_input} — Historical Close Price",font=dict(size=13,color="rgba(255,255,255,0.7)"),x=0),
        xaxis=dict(**PL["xaxis"],type="date",rangeselector=RS,rangeslider=RL))
    st.plotly_chart(fig_h,use_container_width=True)

    st.divider()
    with st.expander("🏢 Peer Comparison — click to expand"):
        @st.cache_data(ttl=3600)
        def get_peers(tk):
            try:
                inf=yf.Ticker(tk).info; sec=inf.get("sector","")
                pm={"Technology":["AAPL","MSFT","GOOGL","META","NVDA","AMZN"],
                    "Consumer Cyclical":["AMZN","TSLA","NKE","MCD","SBUX","WMT"],
                    "Consumer Defensive":["WMT","COST","PG","KO","PEP"],
                    "Healthcare":["JNJ","PFE","UNH","ABBV","MRK"],
                    "Financial Services":["JPM","BAC","WFC","GS","MS","V"],
                    "Energy":["XOM","CVX","COP","SLB","EOG"],
                    "Industrials":["HON","GE","MMM","CAT","BA"],
                    "Communication Services":["GOOGL","META","NFLX","DIS","CMCSA"]}
                return [p for p in pm.get(sec,[]) if p!=tk][:5],sec
            except: return [],""
        peers,sector=get_peers(ticker_input)
        if sector: st.caption(f"Sector: {sector}")
        if peers:
            @st.cache_data(ttl=3600)
            def peer_metrics(tks):
                rows=[]
                for t in tks:
                    try:
                        i2=yf.Ticker(t).info
                        rows.append({"Ticker":t,
                                     "Price":i2.get("currentPrice") or i2.get("regularMarketPrice"),
                                     "P/E":i2.get("trailingPE"),
                                     "ROE %":(i2.get("returnOnEquity") or 0)*100 or None,
                                     "D/E":i2.get("debtToEquity"),
                                     "Mkt Cap $B":(i2.get("marketCap") or 0)/1e9 or None})
                    except: pass
                return rows
            with st.spinner("Fetching peers..."):
                rows=peer_metrics(tuple([ticker_input]+peers))
            if rows:
                df_p=pd.DataFrame(rows).set_index("Ticker")
                st.dataframe(df_p.style.format(na_rep="N/A",precision=2),use_container_width=True)
                vals=df_p["ROE %"].dropna()
                fig_pr=go.Figure(go.Bar(x=vals.index,y=vals.values,
                    marker_color=["#22c55e" if t==ticker_input else "#3b82f6" for t in vals.index],
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
                fig_pr.update_layout(**PL,height=220,showlegend=False,
                    title=dict(text=f"ROE % vs Peers  |  🟩 = {ticker_input}",
                               font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0))
                st.plotly_chart(fig_pr,use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  RATIOS
# ══════════════════════════════════════════════════════════════
with t1:
    st.markdown(f"## Financial Ratios — {ticker_input}")
    st.caption("Most recent annual financial statements via Yahoo Finance")
    def hlth(k,v):
        if v is None: return "⚪"
        bm={"Gross Profit Margin":(40,20),"Operating Profit Margin":(15,5),"Net Profit Margin":(10,3),
            "Return on Assets (ROA)":(10,3),"Return on Equity (ROE)":(15,5),
            "Current Ratio":(1.5,1.0),"Quick Ratio":(1.0,0.5),"Cash Ratio":(0.5,0.2)}
        if k=="Debt-to-Equity": return "🟢" if v<1 else("🟡" if v<3 else "🔴")
        if k in bm:
            hi,lo=bm[k]; return "🟢" if v>=hi else("🟡" if v>=lo else "🔴")
        return "⚪"

    st.markdown("#### 💰 Profitability")
    pc=st.columns(3)
    for i,k in enumerate(["Gross Profit Margin","Operating Profit Margin","Net Profit Margin",
                           "Return on Assets (ROA)","Return on Equity (ROE)","EPS"]):
        v=ratios.get(k); sfx="%" if k!="EPS" else ""
        pc[i%3].metric(f"{hlth(k,v)} {k}",f"{v:.2f}{sfx}" if v is not None else "N/A")

    st.markdown("#### 💧 Liquidity")
    lc=st.columns(3)
    for i,k in enumerate(["Current Ratio","Quick Ratio","Cash Ratio"]):
        v=ratios.get(k); lc[i].metric(f"{hlth(k,v)} {k}",f"{v:.2f}" if v is not None else "N/A")

    st.markdown("#### 📌 Other")
    ac=st.columns(4)
    for i,k in enumerate(["P/E Ratio","Dividend Payout Ratio","Debt-to-Equity","Sustainable Growth Rate"]):
        v=ratios.get(k)
        if v is not None and k in ["Dividend Payout Ratio","Sustainable Growth Rate"]:
            ac[i].metric(f"{hlth(k,v)} {k}",f"{v*100:.1f}%" if v<10 else f"{v:.1f}%")
        else:
            ac[i].metric(f"{hlth(k,v)} {k}",f"{v:.2f}" if v is not None else "N/A")

    fcf=ratios.get("Free Cash Flow")
    if fcf:
        clr="#22c55e" if fcf>0 else "#ef4444"
        st.markdown("#### 💵 Free Cash Flow")
        st.markdown(f"<span style='font-family:DM Mono,monospace;font-size:22px;color:{clr};font-weight:500'>${fcf/1e9:.2f}B</span>",unsafe_allow_html=True)
        st.success("✅ Positive FCF") if fcf>0 else st.warning("⚠️ Negative FCF")
    st.caption("🟢 Strong  🟡 Acceptable  🔴 Weak  ⚪ No benchmark")

# ══════════════════════════════════════════════════════════════
#  CAPM & VALUATION
# ══════════════════════════════════════════════════════════════
with t2:
    st.markdown(f"## CAPM & Valuation — {ticker_input}")
    cl2,cr2=st.columns(2)
    with cl2:
        st.markdown("### Beta & CAPM")
        st.markdown("> **Re = Rf + β × (Rm − Rf)**")
        if beta:
            b1,b2,b3=st.columns(3)
            b1.metric("Beta (β)",f"{beta:.3f}"); b2.metric("R²",f"{r2:.3f}" if r2 else "N/A")
            b3.metric("CAPM Return",f"{re*100:.2f}%" if re else "N/A")
            st.markdown(f"- **Rf:** {rf*100:.2f}%  ·  **Rm:** {rm*100:.2f}%  ·  **β:** {beta:.3f}")
            if beta<0.8: st.success(f"β {beta:.2f} — Less volatile than market")
            elif beta<1.2: st.info(f"β {beta:.2f} — Moves in line with market")
            else: st.warning(f"β {beta:.2f} — More volatile than market")
            if df_reg is not None:
                xl=np.linspace(df_reg["m"].min(),df_reg["m"].max(),100)
                yl=beta*xl+(re/52-beta*rm/52 if re else 0)
                fig_b=go.Figure()
                fig_b.add_trace(go.Scatter(x=df_reg["m"],y=df_reg["s"],mode="markers",
                    marker=dict(color="rgba(59,130,246,0.4)",size=5),name="Weekly returns",
                    hovertemplate="Market: %{x:.3f}<br>Stock: %{y:.3f}<extra></extra>"))
                fig_b.add_trace(go.Scatter(x=xl,y=yl,mode="lines",
                    line=dict(color="#ef4444",width=2),name=f"β={beta:.3f}"))
                fig_b.update_layout(**PL,height=300,
                    title=dict(text=f"Beta Regression (R²={r2:.3f})",font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0))
                st.plotly_chart(fig_b,use_container_width=True)
        else: st.warning("Beta could not be calculated.")

    with cr2:
        st.markdown("### WACC & Intrinsic Value")
        if wacc:
            w1,w2,w3=st.columns(3)
            w1.metric("WACC",f"{wacc*100:.2f}%")
            w2.metric("Cost of Equity",f"{re*100:.2f}%" if re else "N/A")
            w3.metric("Cost of Debt",f"{rd*100:.2f}%" if rd else "N/A")
        else: st.info("WACC could not be calculated.")
        st.divider()
        st.markdown("### DCF Intrinsic Value")
        if intrinsic:
            gap=(intrinsic-cur)/cur*100
            i1,i2,i3=st.columns(3)
            i1.metric("Intrinsic Value",f"${intrinsic:.2f}")
            i2.metric("Market Price",f"${cur:.2f}")
            i3.metric("Gap",f"{gap:+.1f}%")
            if gap>15: st.success(f"✅ Potentially undervalued by {gap:.1f}%")
            elif gap<-15: st.warning(f"⚠️ Potentially overvalued by {abs(gap):.1f}%")
            else: st.info(f"ℹ️ Fairly valued (gap {gap:+.1f}%)")
            st.caption("Gordon Growth Model · 3% terminal growth rate")
        else: st.info("Intrinsic value could not be calculated.")

# ══════════════════════════════════════════════════════════════
#  MONTE CARLO
# ══════════════════════════════════════════════════════════════
with t3:
    st.markdown(f"## Monte Carlo Simulation — {ticker_input}")
    st.caption(f"{iterations:,} simulations × {prediction_days} trading days · Target: {tgt}")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Start Price",f"${S0:.2f}"); m2.metric("Median",f"${p50:.2f}",f"{(p50-S0)/S0*100:+.1f}%")
    m3.metric("25th Pct",f"${p25:.2f}"); m4.metric("75th Pct",f"${p75:.2f}"); m5.metric("5th Pct",f"${p5:.2f}")

    fdates=pd.date_range(start=hist.index[-1],periods=prediction_days,freq="B")
    fig_mc=go.Figure()
    for i in range(min(80,spaths.shape[1])):
        fig_mc.add_trace(go.Scatter(x=fdates,y=spaths[:,i],mode="lines",
            line=dict(color="rgba(59,130,246,0.07)",width=0.5),showlegend=False,hoverinfo="skip"))
    fig_mc.add_trace(go.Scatter(x=list(fdates)+list(fdates[::-1]),y=list(p95p)+list(p5p[::-1]),
        fill="toself",fillcolor="rgba(239,68,68,0.07)",line=dict(color="rgba(0,0,0,0)"),name="90% CI",hoverinfo="skip"))
    fig_mc.add_trace(go.Scatter(x=list(fdates)+list(fdates[::-1]),y=list(p75p)+list(p25p[::-1]),
        fill="toself",fillcolor="rgba(245,158,11,0.1)",line=dict(color="rgba(0,0,0,0)"),name="50% CI",hoverinfo="skip"))
    for yv,nm,clr in [(p95p,f"95th ${p95:.0f}","#ef4444"),(p5p,f"5th ${p5:.0f}","#ef4444"),
                       (p75p,f"75th ${p75:.0f}","#f59e0b"),(p25p,f"25th ${p25:.0f}","#f59e0b"),
                       (p50p,f"Median ${p50:.0f}","#22c55e")]:
        fig_mc.add_trace(go.Scatter(x=fdates,y=yv,mode="lines",name=nm,
            line=dict(color=clr,width=2 if "Median" in nm else 1,dash="solid" if "Median" in nm else "dash"),
            hovertemplate=f"{nm}<br>%{{x|%b %d, %Y}}: $%{{y:.2f}}<extra></extra>"))
    fig_mc.add_hline(y=S0,line_color="white",line_width=1,opacity=0.4,
                     annotation_text=f"Start ${S0:.2f}",annotation_font_color="rgba(255,255,255,0.5)")
    fig_mc.update_layout(**PL,height=420,
        title=dict(text=f"{ticker_input} — {prediction_days}-Day Monte Carlo Forecast",
                   font=dict(size=13,color="rgba(255,255,255,0.7)"),x=0),
        xaxis=dict(**PL["xaxis"],type="date",rangeselector=RS,rangeslider=RL))
    st.plotly_chart(fig_mc,use_container_width=True)

    fig_dist=go.Figure()
    fig_dist.add_trace(go.Histogram(x=finals,nbinsx=60,marker_color="rgba(59,130,246,0.7)",
        hovertemplate="$%{x:.0f}: %{y} paths<extra></extra>"))
    for v,n,c in [(p5,"5th","#ef4444"),(p25,"25th","#f59e0b"),(p50,"Median","#22c55e"),
                  (p75,"75th","#f59e0b"),(p95,"95th","#ef4444"),(S0,"Start","#fff")]:
        fig_dist.add_vline(x=v,line_color=c,line_width=1.5,
                           annotation_text=f"{n} ${v:.0f}",
                           annotation_font_color=c,annotation_font_size=10)
    fig_dist.update_layout(**PL,height=260,showlegend=False,
        title=dict(text="Distribution of Final Prices",font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0))
    st.plotly_chart(fig_dist,use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PROPHET
# ══════════════════════════════════════════════════════════════
with t4:
    st.markdown(f"## Prophet Forecast — {ticker_input}")
    cs=hist["Close"].squeeze().dropna().reset_index()
    cs.columns=["ds","y"]; cs["ds"]=pd.to_datetime(cs["ds"]).dt.tz_localize(None)
    fdf=None; meth=None

    if PROPHET_AVAILABLE:
        st.caption("Facebook Prophet — seasonality-aware time series forecasting")
        with st.spinner("Running Prophet..."):
            try:
                m=Prophet(daily_seasonality=False,yearly_seasonality=True,
                          weekly_seasonality=True,changepoint_prior_scale=0.05)
                m.fit(cs); future=m.make_future_dataframe(periods=prediction_days)
                fdf=m.predict(future); meth="Prophet"
            except Exception as e: st.warning(f"Prophet failed: {e}")
    else:
        st.info("Prophet not installed. Run: `pip install prophet`")

    if fdf is None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with st.spinner("Running Exp. Smoothing fallback..."):
                model=ExponentialSmoothing(cs["y"].values,trend="add",seasonal=None).fit()
                fv=model.forecast(prediction_days)
                ld=cs["ds"].iloc[-1]
                fd=pd.date_range(start=ld+timedelta(days=1),periods=prediction_days,freq="D")
                hf=cs.rename(columns={"y":"yhat"}); hf["yhat_lower"]=hf["yhat"]; hf["yhat_upper"]=hf["yhat"]
                ff=pd.DataFrame({"ds":fd,"yhat":fv,"yhat_lower":fv*0.9,"yhat_upper":fv*1.1})
                fdf=pd.concat([hf,ff],ignore_index=True); meth="Exp. Smoothing"
        except: st.error("Forecast unavailable. Install `prophet` or `statsmodels`.")

    if fdf is not None:
        hl=len(cs); fc_fut=fdf.iloc[hl:]
        fend=float(fc_fut["yhat"].iloc[-1]) if not fc_fut.empty else None
        fup=float(fc_fut["yhat_upper"].iloc[-1]) if not fc_fut.empty else None
        flo=float(fc_fut["yhat_lower"].iloc[-1]) if not fc_fut.empty else None
        if fend:
            fchg=(fend-cur)/cur*100
            f1,f2,f3,f4=st.columns(4)
            f1.metric("Current Price",f"${cur:.2f}")
            f2.metric(f"{meth} Forecast",f"${fend:.2f}",f"{fchg:+.1f}%")
            f3.metric("Upper Bound",f"${fup:.2f}" if fup else "N/A")
            f4.metric("Lower Bound",f"${flo:.2f}" if flo else "N/A")

        fig_p=go.Figure()
        fig_p.add_trace(go.Scatter(x=cs["ds"],y=cs["y"],mode="lines",name="Historical",
            line=dict(color="#3b82f6",width=1.5),
            hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
        if not fc_fut.empty:
            fig_p.add_trace(go.Scatter(
                x=pd.concat([fc_fut["ds"],fc_fut["ds"][::-1]]),
                y=pd.concat([fc_fut["yhat_upper"],fc_fut["yhat_lower"][::-1]]),
                fill="toself",fillcolor="rgba(34,197,94,0.07)",
                line=dict(color="rgba(0,0,0,0)"),name="CI",hoverinfo="skip"))
            fig_p.add_trace(go.Scatter(x=fc_fut["ds"],y=fc_fut["yhat"],mode="lines",
                name=f"{meth} Forecast",line=dict(color="#22c55e",width=2,dash="dot"),
                hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
        fig_p.add_vline(x=str(cs["ds"].iloc[-1]),line_color="rgba(255,255,255,0.2)",line_dash="dash",
                        annotation_text="Today",annotation_font_color="rgba(255,255,255,0.3)")
        fig_p.update_layout(**PL,height=400,
            title=dict(text=f"{ticker_input} — {meth} Price Forecast ({prediction_days}d)",
                       font=dict(size=13,color="rgba(255,255,255,0.7)"),x=0),
            xaxis=dict(**PL["xaxis"],type="date",rangeselector=RS,rangeslider=RL))
        st.plotly_chart(fig_p,use_container_width=True)

        if fend:
            st.markdown("#### vs Monte Carlo")
            c1,c2,c3=st.columns(3)
            c1.metric("MC Median",f"${p50:.2f}",f"{(p50-cur)/cur*100:+.1f}%")
            c2.metric(meth,f"${fend:.2f}",f"{fchg:+.1f}%")
            avg=(p50+fend)/2
            c3.metric("Average",f"${avg:.2f}",f"{(avg-cur)/cur*100:+.1f}%")
            diff=abs(p50-fend)/cur
            if diff<0.05: st.success("✅ Strong agreement between MC and Prophet.")
            elif diff<0.15: st.info("ℹ️ Moderate divergence between MC and Prophet.")
            else: st.warning("⚠️ Significant divergence — interpret with caution.")

        if meth=="Prophet" and "trend" in fdf.columns:
            st.divider()
            st.markdown("#### Seasonality Components")
            st.caption("Prophet decomposes forecast into trend, yearly, and weekly components")
            cd=fdf[["ds","trend","yearly","weekly"]].dropna()
            fig_c=make_subplots(rows=3,cols=1,
                subplot_titles=("Trend","Yearly Seasonality","Weekly Seasonality"),
                vertical_spacing=0.08)
            for row,col,clr in [(1,"trend","#3b82f6"),(2,"yearly","#22c55e"),(3,"weekly","#f59e0b")]:
                if col in cd.columns:
                    fig_c.add_trace(go.Scatter(x=cd["ds"],y=cd[col],mode="lines",
                        line=dict(color=clr,width=1.5),name=col,
                        hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>"),row=row,col=1)
            fig_c.update_layout(**PL,height=480,showlegend=False)
            fig_c.update_xaxes(gridcolor="rgba(255,255,255,0.05)",showline=False)
            fig_c.update_yaxes(gridcolor="rgba(255,255,255,0.05)",showline=False,zeroline=False)
            st.plotly_chart(fig_c,use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with t5:
    st.markdown(f"## Technical Analysis — {ticker_input}")
    st.caption(f"MA periods auto-adjusted: **{mf} / {ms}** · {horizon} · {risk_tolerance} risk")

    cl1y=hist["Close"].squeeze()
    cl1y=cl1y[cl1y.index>=(cl1y.index[-1]-pd.DateOffset(years=1))]
    px_=cl1y.values; td=cl1y.index

    s20=_sma(px_,mf); s50=_sma(px_,ms)
    e20=_ema(px_,mf); e50=_ema(px_,ms)
    w20=_wma(px_,mf); w50=_wma(px_,ms)
    sb,sbe=_crosses(s20,s50); eb,ebe=_crosses(e20,e50); wb,wbe=_crosses(w20,w50)
    bbs=pd.Series(px_).rolling(mf).std().values
    bbu=s20+2*bbs; bbl=s20-2*bbs
    rsi_=_rsi(px_); e12=_ema(px_,12); e26=_ema(px_,26)
    ml=e12-e26; sl=_ema(ml,9); mh=ml-sl
    rv=rsi_[-1]; lp=px_[-1]

    bsig,bsig2=0,0
    if s20[-1]>s50[-1] and e20[-1]>e50[-1] and w20[-1]>w50[-1]: bsig+=1
    else: bsig2+=1
    if rv<50: bsig2+=1
    elif rv>55: bsig+=1
    if ml[-1]>sl[-1]: bsig+=1
    else: bsig2+=1
    if lp<bbl[-1]: bsig+=1
    elif lp>bbu[-1]: bsig2+=1

    if bsig>=4: tc2,ts2,tt2,tw2="#22c55e","STRONG BUY","Most indicators bullish","Technicals support entry."
    elif bsig>=3: tc2,ts2,tt2,tw2="#22c55e","BUY","More bullish than bearish","Consider entering in stages."
    elif bsig2>=4: tc2,ts2,tt2,tw2="#ef4444","WAIT","Multiple bearish signals",f"Watch for RSI<40 + MACD cross."
    elif bsig2>=3: tc2,ts2,tt2,tw2="#f59e0b","CAUTION","Mixed, leaning bearish","Wait for clearer signal."
    else: tc2,ts2,tt2,tw2="#3b82f6","NEUTRAL","No dominant signal","Watch for consensus MA crossover."

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);border:2px solid {tc2};border-radius:12px;
                padding:16px 20px;margin-bottom:20px">
        <div style="font-size:20px;font-weight:600;color:{tc2};margin-bottom:6px">{ts2} — {tt2}</div>
        <div style="font-size:13px;color:{tc2};opacity:0.8"><b>What to watch:</b> {tw2}</div>
    </div>
    """,unsafe_allow_html=True)

    def mk_cross(fig,bulls,bears,dates,prices,lbl):
        for i in bulls:
            if i<len(dates):
                fig.add_trace(go.Scatter(x=[dates[i]],y=[prices[i]],mode="markers",
                    marker=dict(symbol="triangle-up",size=12,color="#22c55e",line=dict(color="#080c14",width=1)),
                    name="▲ Golden Cross",showlegend=(i==bulls[0]),
                    hovertemplate=f"▲ Golden Cross<br>%{{x|%b %d, %Y}}: $%{{y:.2f}}<extra></extra>"))
        for i in bears:
            if i<len(dates):
                fig.add_trace(go.Scatter(x=[dates[i]],y=[prices[i]],mode="markers",
                    marker=dict(symbol="triangle-down",size=12,color="#ef4444",line=dict(color="#080c14",width=1)),
                    name="▼ Death Cross",showlegend=(i==bears[0]),
                    hovertemplate=f"▼ Death Cross<br>%{{x|%b %d, %Y}}: $%{{y:.2f}}<extra></extra>"))

    it0,it1,it2,it3,it4=st.tabs([f"SMA {mf}/{ms}",f"EMA {mf}/{ms}",f"WMA {mf}/{ms}","RSI","MACD"])

    for tab_,fast,slow,bulls,bears,c1_,c2_,lbl_ in [
        (it0,s20,s50,sb,sbe,"#f59e0b","#ef4444",f"SMA {mf}/{ms}"),
        (it1,e20,e50,eb,ebe,"#22c55e","#1abc9c",f"EMA {mf}/{ms}"),
        (it2,w20,w50,wb,wbe,"#9b59b6","#8e44ad",f"WMA {mf}/{ms}"),
    ]:
        with tab_:
            up=fast[-1]>slow[-1]
            sig_txt=f"🟢 {lbl_.split('/')[0].split()[0]}{mf} above {lbl_.split('/')[0].split()[0]}{ms} — uptrend" if up else f"🔴 {lbl_.split('/')[0].split()[0]}{mf} below {lbl_.split('/')[0].split()[0]}{ms} — downtrend"
            st.markdown(f"**Current signal:** {sig_txt}")
            fig_=go.Figure()
            fig_.add_trace(go.Scatter(x=td,y=px_,mode="lines",name="Price",
                line=dict(color="#3b82f6",width=1.5),
                hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
            fig_.add_trace(go.Scatter(x=td,y=fast,mode="lines",name=f"{lbl_.split('/')[0]}",
                line=dict(color=c1_,width=1.5,dash="dash"),
                hovertemplate=f"Fast: $%{{y:.2f}}<extra></extra>"))
            fig_.add_trace(go.Scatter(x=td,y=slow,mode="lines",name=f"{lbl_.split('/')[1] if '/' in lbl_ else lbl_}",
                line=dict(color=c2_,width=1.5,dash="dash"),
                hovertemplate=f"Slow: $%{{y:.2f}}<extra></extra>"))
            mk_cross(fig_,bulls,bears,td,px_,lbl_)
            fig_.update_layout(**PL,height=380,
                title=dict(text=f"{lbl_} — ▲ Golden Cross  ▼ Death Cross",
                           font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0),
                xaxis=dict(**PL["xaxis"],type="date",rangeselector=RS,rangeslider=RL))
            st.plotly_chart(fig_,use_container_width=True)
            with st.expander("❓ How to read this indicator"):
                desc={"SMA":"Simple average of last N prices. Slow to react — great for confirming trends.",
                      "EMA":"Weights recent prices more. Faster signal but more false positives.",
                      "WMA":"Linear weighting — sits between SMA and EMA in speed."}
                key=lbl_.split()[0]
                st.markdown(f"{desc.get(key,'')} **▲ Golden Cross** = fast MA crosses above slow = bullish. **▼ Death Cross** = fast below slow = bearish.")

    with it3:
        rc="#ef4444" if rv>70 else("#22c55e" if rv<30 else "#f59e0b")
        rl="Overbought" if rv>70 else("Oversold" if rv<30 else "Neutral")
        st.markdown(f"**Current RSI:** <span style='color:{rc};font-weight:600;font-size:16px'>{rv:.1f} — {rl}</span>",unsafe_allow_html=True)
        fig_r=go.Figure()
        fig_r.add_hrect(y0=70,y1=100,fillcolor="rgba(239,68,68,0.06)",line_width=0)
        fig_r.add_hrect(y0=0,y1=30,fillcolor="rgba(34,197,94,0.06)",line_width=0)
        fig_r.add_trace(go.Scatter(x=td,y=rsi_,mode="lines",name="RSI (14)",
            line=dict(color="#3b82f6",width=1.5),
            hovertemplate="%{x|%b %d, %Y}: %{y:.1f}<extra></extra>"))
        fig_r.add_hline(y=70,line_color="#ef4444",line_width=1,line_dash="dash",
                        annotation_text="Overbought 70",annotation_font_color="#ef4444",annotation_font_size=10)
        fig_r.add_hline(y=30,line_color="#22c55e",line_width=1,line_dash="dash",
                        annotation_text="Oversold 30",annotation_font_color="#22c55e",annotation_font_size=10)
        fig_r.update_layout(**PL,height=320,yaxis=dict(**PL["yaxis"],range=[0,100]),
            title=dict(text="RSI — Relative Strength Index (14-day)",font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0),
            xaxis=dict(**PL["xaxis"],type="date",rangeselector=RS,rangeslider=RL))
        st.plotly_chart(fig_r,use_container_width=True)
        with st.expander("❓ How to read RSI"):
            st.markdown("**>70** = overbought, pullback likely. **<30** = oversold, bounce likely. **40–60** = neutral. Rising above 50 = building momentum.")

    with it4:
        ms_="🟢 MACD above signal — positive momentum" if ml[-1]>sl[-1] else "🔴 MACD below signal — negative momentum"
        st.markdown(f"**Current signal:** {ms_}")
        fig_m=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.06,row_heights=[0.55,0.45])
        fig_m.add_trace(go.Scatter(x=td,y=px_,mode="lines",name="Price",
            line=dict(color="#3b82f6",width=1.5),
            hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"),row=1,col=1)
        fig_m.add_trace(go.Scatter(x=td,y=ml,mode="lines",name="MACD",
            line=dict(color="#3b82f6",width=1.5),
            hovertemplate="MACD: %{y:.3f}<extra></extra>"),row=2,col=1)
        fig_m.add_trace(go.Scatter(x=td,y=sl,mode="lines",name="Signal",
            line=dict(color="#f59e0b",width=1.5),
            hovertemplate="Signal: %{y:.3f}<extra></extra>"),row=2,col=1)
        bc_=["rgba(34,197,94,0.6)" if v>=0 else "rgba(239,68,68,0.6)" for v in mh]
        fig_m.add_trace(go.Bar(x=td,y=mh,name="Histogram",marker_color=bc_,
            hovertemplate="%{x|%b %d, %Y}: %{y:.3f}<extra></extra>"),row=2,col=1)
        fig_m.update_layout(**PL,height=440,
            title=dict(text="MACD (12, 26, 9)",font=dict(size=12,color="rgba(255,255,255,0.6)"),x=0),
            xaxis2=dict(gridcolor="rgba(255,255,255,0.05)",type="date",rangeselector=RS,rangeslider=RL))
        fig_m.update_yaxes(gridcolor="rgba(255,255,255,0.05)",showline=False,zeroline=False)
        st.plotly_chart(fig_m,use_container_width=True)
        with st.expander("❓ How to read MACD"):
            st.markdown("**MACD crosses above signal** = bullish momentum. **MACD crosses below** = momentum turning negative. Green histogram = accelerating up. Red = accelerating down.")

# ══════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════
with t6:
    st.markdown("## About AlphaView")
    st.markdown("""
    ### How It Works
    AlphaView combines 4 data signals to generate scenarios:

    | Signal | Source |
    |--------|--------|
    | Monte Carlo Direction | Median of simulated price paths |
    | CAPM Expected Return | Risk-adjusted return vs market |
    | Ratio Health | Profitability + liquidity ratios |
    | Valuation Gap | Intrinsic value vs market price |

    ### Dynamic MA Periods
    Moving average periods auto-adjust for your profile:
    - Short-term + High risk → faster, tighter windows (e.g. 8/16)
    - Long-term + Low risk → slower, wider windows (e.g. 60/240)

    ### Data Sources
    - Prices & fundamentals: Yahoo Finance via `yfinance`
    - Risk-free rate: Live 10-yr US Treasury (`^TNX`)
    - Market return: S&P 500 10-yr average (`^GSPC`)
    - Forecast: Facebook Prophet (Exp. Smoothing fallback)

    ### ⚠️ Disclaimer
    > Educational purposes only. Not financial advice.
    > Always consult a qualified advisor before investing.

    ---
    *DAB401 Final Project · St. Clair College*
    """)

st.markdown("""
<div style="margin-top:40px;padding:12px 0;border-top:1px solid rgba(255,255,255,0.05);
            display:flex;justify-content:space-between">
    <span style="font-size:11px;color:rgba(255,255,255,0.2)">Data via Yahoo Finance · Prices delayed</span>
    <span style="font-size:11px;color:rgba(255,255,255,0.2)">AlphaView · DAB401 · St. Clair College</span>
</div>
""",unsafe_allow_html=True)
