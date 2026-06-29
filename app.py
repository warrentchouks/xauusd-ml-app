import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="XAUUSD ML Scanner", page_icon="🥇", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0d1117; }
h1, h2, h3 { color: #ffffff !important; }
p, div, span, label { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

def is_market_open():
    now = datetime.now(timezone.utc)
    d = now.weekday()
    h = now.hour
    if d == 6 and h < 23:
        return False, "Dimanche — Ouvre a 23h00 UTC"
    if d == 5:
        return False, "Samedi — Marche ferme"
    if d == 4 and h >= 22:
        return False, "Weekend — Marche ferme"
    return True, "Marche ouvert"

def download_data(ticker, interval="1h", period="60d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    except:
        return None

def resample_tf(df, tf):
    return df[["Open", "High", "Low", "Close", "Volume"]].resample(tf).agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()

def get_position(proba, capital, atr, prix, kf=0.126, rr=2.0):
    if proba >= 0.85:
        r = min(kf * 2.0, 0.03)
    elif proba >= 0.80:
        r = min(kf * 1.5, 0.025)
    elif proba >= 0.75:
        r = min(kf * 1.0, 0.02)
    elif proba >= 0.70:
        r = min(kf * 0.75, 0.015)
    else:
        r = min(kf * 0.5, 0.01)
    sl = 1.5 * atr
    return {
        "risk_pct": r * 100,
        "risk_eur": capital * r,
        "sl_long": prix - sl,
        "tp_long": prix + sl * rr,
        "sl_short": prix + sl,
        "tp_short": prix - sl * rr
    }

@st.cache_resource
def load_models():
    try:
        mx  = joblib.load("model_xgb_final.pkl")
        ml  = joblib.load("model_lgb_final.pkl")
        mx2 = joblib.load("model_xgb2_final.pkl")
        ft  = joblib.load("features_final.pkl")
        with open("config.json") as f:
            cfg = json.load(f)
        return mx, ml, mx2, ft, cfg
    except Exception as e:
        st.error("Erreur modeles: " + str(e))
        return None, None, None, None, None

@st.cache_data(ttl=900)
def get_all_data():
    actifs = {
        "GOLD": "GC=F", "DXY": "DX-Y.NYB", "VIX": "^VIX",
        "US10Y": "^TNX", "SILVER": "SI=F", "SP500": "^GSPC"
    }
    h1 = {}
    for nom, t in actifs.items():
        df = download_data(t)
        if df is not None and len(df) > 50:
            h1[nom] = df
    d1 = download_data("GC=F", "1d", "5y")
    return h1, d1

@st.cache_data(ttl=900)
def build_features(_h1, _d1, _ft):
    try:
        if "GOLD" not in _h1 or _d1 is None:
            return None
        df = _h1["GOLD"].copy()
        for nom in ["DXY", "VIX", "US10Y", "SILVER", "SP500"]:
            if nom in _h1:
                tmp = _h1[nom][["Close"]].rename(columns={"Close": nom})
                df = pd.merge_asof(df.sort_index(), tmp.sort_index(),
                                   left_index=True, right_index=True,
                                   direction="backward")
        df = df.ffill().dropna()
        if len(df) < 100:
            return None
        h4 = resample_tf(df, "4h")
        h4["EMA20_H4"] = ta.trend.ema_indicator(h4["Close"], 20)
        h4["EMA50_H4"] = ta.trend.ema_indicator(h4["Close"], 50)
        h4["RSI_H4"]   = ta.momentum.rsi(h4["Close"], 14)
        h4["Trend_H4"] = (h4["EMA20_H4"] > h4["EMA50_H4"]).astype(int)
        h4 = h4.dropna()
        d1 = _d1.copy()
        d1["EMA20_D1"]  = ta.trend.ema_indicator(d1["Close"], 20)
        d1["EMA50_D1"]  = ta.trend.ema_indicator(d1["Close"], 50)
        d1["EMA200_D1"] = ta.trend.ema_indicator(d1["Close"], 200)
        d1["RSI_D1"]    = ta.momentum.rsi(d1["Close"], 14)
        d1["Trend_D1"]  = (d1["EMA20_D1"] > d1["EMA50_D1"]).astype(int)
        d1 = d1.dropna()
        df = pd.merge_asof(df.sort_index(),
            h4[["Trend_H4", "EMA20_H4", "EMA50_H4", "RSI_H4"]].sort_index(),
            left_index=True, right_index=True, direction="backward")
        df = pd.merge_asof(df.sort_index(),
            d1[["Trend_D1", "EMA20_D1", "EMA50_D1", "EMA200_D1", "RSI_D1"]].sort_index(),
            left_index=True, right_index=True, direction="backward")
        df = df.ffill().dropna()
        if len(df) < 50:
            return None
        df["RSI"]    = ta.momentum.rsi(df["Close"], 14)
        df["RSI_ob"] = (df["RSI"] > 70).astype(int)
        df["RSI_os"] = (df["RSI"] < 30).astype(int)
        df["EMA20"]  = ta.trend.ema_indicator(df["Close"], 20)
        df["EMA50"]  = ta.trend.ema_indicator(df["Close"], 50)
        df["EMA200"] = ta.trend.ema_indicator(df["Close"], 200)
        macd = ta.trend.MACD(df["Close"])
        df["MACD"]            = macd.macd()
        df["MACD_signal"]     = macd.macd_signal()
        df["MACD_hist"]       = macd.macd_diff()
        df["MACD_cross_up"]   = ((df["MACD"] > df["MACD_signal"]) &
            (df["MACD"].shift(1) < df["MACD_signal"].shift(1))).astype(int)
        df["MACD_cross_down"] = ((df["MACD"] < df["MACD_signal"]) &
            (df["MACD"].shift(1) > df["MACD_signal"].shift(1))).astype(int)
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = df["BB_upper"] - df["BB_lower"]
        df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / df["BB_width"]
        df["ATR"]      = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], 14)
        df["Return_1h"]  = df["Close"].pct_change(1)
        df["Return_4h"]  = df["Close"].pct_change(4)
        df["Return_24h"] = df["Close"].pct_change(24)
        df["Price_vs_EMA20"]  = (df["Close"] - df["EMA20"]) / df["EMA20"] * 100
        df["Price_vs_EMA50"]  = (df["Close"] - df["EMA50"]) / df["EMA50"] * 100
        df["Price_vs_EMA200"] = (df["Close"] - df["EMA200"]) / df["EMA200"] * 100
        df["EMA20_vs_EMA50"]  = (df["EMA20"] - df["EMA50"]) / df["EMA50"] * 100
        df["High_Low_pct"]    = (df["High"] - df["Low"]) / df["Close"] * 100
        df["Hour"]            = df.index.hour
        df["Session_London"]  = ((df["Hour"] >= 7) & (df["Hour"] < 16)).astype(int)
        df["Session_NY"]      = ((df["Hour"] >= 13) & (df["Hour"] < 21)).astype(int)
        df["Session_Overlap"] = ((df["Hour"] >= 13) & (df["Hour"] < 16)).astype(int)
        df["Session_Asian"]   = ((df["Hour"] >= 0) & (df["Hour"] < 7)).astype(int)
        df["DXY_return"]    = df["DXY"].pct_change(4) if "DXY" in df.columns else 0
        df["VIX_return"]    = df["VIX"].pct_change(4) if "VIX" in df.columns else 0
        df["SILVER_return"] = df["SILVER"].pct_change(4) if "SILVER" in df.columns else 0
        df["SP500_return"]  = df["SP500"].pct_change(4) if "SP500" in df.columns else 0
        df["Gold_DXY_div"]   = df["Return_4h"] + df["DXY_return"]
        df["Trend_bull_3TF"] = ((df["Trend_D1"] == 1) & (df["Trend_H4"] == 1)).astype(int)
        df["Trend_bear_3TF"] = ((df["Trend_D1"] == 0) & (df["Trend_H4"] == 0)).astype(int)
        df["BOS_bull"] = (df["High"] > df["High"].shift(1).rolling(10).max()).astype(int)
        df["BOS_bear"] = (df["Low"] < df["Low"].shift(1).rolling(10).min()).astype(int)
        df["Higher_High"] = (df["High"] > df["High"].shift(1)).astype(int)
        df["Lower_Low"]   = (df["Low"] < df["Low"].shift(1)).astype(int)
        df["CHoCH_bull"]  = ((df["Lower_Low"].shift(3) == 1) &
            (df["Lower_Low"].shift(2) == 1) & (df["Higher_High"] == 1)).astype(int)
        df["CHoCH_bear"]  = ((df["Higher_High"].shift(3) == 1) &
            (df["Higher_High"].shift(2) == 1) & (df["Lower_Low"] == 1)).astype(int)
        df["Bearish_candle"] = (df["Close"] < df["Open"]).astype(int)
        df["Bullish_candle"] = (df["Close"] > df["Open"]).astype(int)
        df["Strong_up"]   = (df["Return_1h"] > df["ATR"] / df["Close"] * 100).astype(int)
        df["Strong_down"] = (df["Return_1h"] < -df["ATR"] / df["Close"] * 100).astype(int)
        df["OB_bull"] = ((df["Bearish_candle"].shift(1) == 1) &
            (df["Strong_up"] == 1)).astype(int)
        df["OB_bear"] = ((df["Bullish_candle"].shift(1) == 1) &
            (df["Strong_down"] == 1)).astype(int)
        df["Range_high"]  = df["High"].rolling(100).max()
        df["Range_low"]   = df["Low"].rolling(100).min()
        df["Range_mid"]   = (df["Range_high"] + df["Range_low"]) / 2
        df["In_discount"] = (df["Close"] < df["Range_mid"]).astype(int)
        df["In_premium"]  = (df["Close"] > df["Range_mid"]).astype(int)
        tol = 0.001
        df["Equal_highs"] = (abs(df["High"] - df["High"].shift(1)) /
            df["High"] < tol).astype(int)
        df["Equal_lows"]  = (abs(df["Low"] - df["Low"].shift(1)) /
            df["Low"] < tol).astype(int)
        df["FVG_bull"]    = ((df["Low"] > df["High"].shift(2)) &
            (df["Low"].shift(1) > df["High"].shift(2))).astype(int)
        df["FVG_bear"]    = ((df["High"] < df["Low"].shift(2)) &
            (df["High"].shift(1) < df["Low"].shift(2))).astype(int)
        df["SMC_bull_score"]   = (df["BOS_bull"] + df["CHoCH_bull"] +
            df["OB_bull"] + df["In_discount"] + df["FVG_bull"])
        df["SMC_bear_score"]   = (df["BOS_bear"] + df["CHoCH_bear"] +
            df["OB_bear"] + df["In_premium"] + df["FVG_bear"])
        df["SMC_bull_confirm"] = ((df["SMC_bull_score"] >= 2) &
            (df["Trend_bull_3TF"] == 1)).astype(int)
        df["SMC_bear_confirm"] = ((df["SMC_bear_score"] >= 2) &
            (df["Trend_bear_3TF"] == 1)).astype(int)
        stoch = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"], 14)
        df["Stoch_K"]          = stoch.stoch()
        df["Stoch_D"]          = stoch.stoch_signal()
        df["Stoch_ob"]         = (df["Stoch_K"] > 80).astype(int)
        df["Stoch_os"]         = (df["Stoch_K"] < 20).astype(int)
        df["Stoch_cross_up"]   = ((df["Stoch_K"] > df["Stoch_D"]) &
            (df["Stoch_K"].shift(1) < df["Stoch_D"].shift(1))).astype(int)
        df["Stoch_cross_down"] = ((df["Stoch_K"] < df["Stoch_D"]) &
            (df["Stoch_K"].shift(1) > df["Stoch_D"].shift(1))).astype(int)
        df["Williams_R"]  = ta.momentum.williams_r(
            df["High"], df["Low"], df["Close"], 14)
        df["Williams_ob"] = (df["Williams_R"] > -20).astype(int)
        df["Williams_os"] = (df["Williams_R"] < -80).astype(int)
        df["CCI"]    = ta.trend.cci(df["High"], df["Low"], df["Close"], 20)
        df["CCI_ob"] = (df["CCI"] > 100).astype(int)
        df["CCI_os"] = (df["CCI"] < -100).astype(int)
        ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"], 9, 26, 52)
        df["Ichi_tenkan"]   = ichi.ichimoku_conversion_line()
        df["Ichi_kijun"]    = ichi.ichimoku_base_line()
        df["Ichi_spanA"]    = ichi.ichimoku_a()
        df["Ichi_spanB"]    = ichi.ichimoku_b()
        df["Ichi_bull"]     = (df["Close"] > df["Ichi_spanA"]).astype(int)
        df["Ichi_tk_cross"] = (df["Ichi_tenkan"] > df["Ichi_kijun"]).astype(int)
        df["VWAP"]          = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["Price_vs_VWAP"] = (df["Close"] - df["VWAP"]) / df["VWAP"] * 100
        df["Above_VWAP"]    = (df["Close"] > df["VWAP"]).astype(int)
        df["OBV"]            = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["OBV_EMA"]        = ta.trend.ema_indicator(df["OBV"], 20)
        df["OBV_trend"]      = (df["OBV"] > df["OBV_EMA"]).astype(int)
        df["OBV_divergence"] = ((df["Close"] > df["Close"].shift(5)) &
            (df["OBV"] < df["OBV"].shift(5))).astype(int)
        pp = resample_tf(df, "1D")
        pp["PP"] = (pp["High"] + pp["Low"] + pp["Close"]) / 3
        pp["R1"] = 2 * pp["PP"] - pp["Low"]
        pp["S1"] = 2 * pp["PP"] - pp["High"]
        pp["R2"] = pp["PP"] + (pp["High"] - pp["Low"])
        pp["S2"] = pp["PP"] - (pp["High"] - pp["Low"])
        pp["R3"] = pp["High"] + 2 * (pp["PP"] - pp["Low"])
        pp["S3"] = pp["Low"] - 2 * (pp["High"] - pp["PP"])
        pv = pp[["PP", "R1", "S1", "R2", "S2", "R3", "S3"]].shift(1)
        df = pd.merge_asof(df.sort_index(), pv.sort_index(),
                           left_index=True, right_index=True, direction="backward")
        df["Dist_PP"]  = (df["Close"] - df["PP"]) / df["Close"] * 100
        df["Dist_R1"]  = (df["R1"] - df["Close"]) / df["Close"] * 100
        df["Dist_S1"]  = (df["Close"] - df["S1"]) / df["Close"] * 100
        df["Near_PP"]  = (abs(df["Dist_PP"]) < 0.1).astype(int)
        df["Near_R1"]  = (abs(df["Dist_R1"]) < 0.1).astype(int)
        df["Near_S1"]  = (abs(df["Dist_S1"]) < 0.1).astype(int)
        df["Above_PP"] = (df["Close"] > df["PP"]).astype(int)
        df["Body"]          = abs(df["Close"] - df["Open"])
        df["Upper_wick"]    = df["High"] - df[["Close", "Open"]].max(axis=1)
        df["Lower_wick"]    = df[["Close", "Open"]].min(axis=1) - df["Low"]
        df["Range_bar"]     = df["High"] - df["Low"]
        df["Doji"]          = (df["Body"] < df["Range_bar"] * 0.1).astype(int)
        df["Hammer"]        = ((df["Lower_wick"] > df["Body"] * 2) &
            (df["Upper_wick"] < df["Body"] * 0.5) &
            (df["Close"] > df["Open"])).astype(int)
        df["Shooting_star"] = ((df["Upper_wick"] > df["Body"] * 2) &
            (df["Lower_wick"] < df["Body"] * 0.5) &
            (df["Close"] < df["Open"])).astype(int)
        df["Bull_engulf"]   = ((df["Close"].shift(1) < df["Open"].shift(1)) &
            (df["Close"] > df["Open"]) &
            (df["Close"] > df["Open"].shift(1)) &
            (df["Open"] < df["Close"].shift(1))).astype(int)
        df["Bear_engulf"]   = ((df["Close"].shift(1) > df["Open"].shift(1)) &
            (df["Close"] < df["Open"]) &
            (df["Close"] < df["Open"].shift(1)) &
            (df["Open"] > df["Close"].shift(1))).astype(int)
        df["Bull_marubozu"] = ((df["Close"] > df["Open"]) &
            (df["Upper_wick"] < df["Body"] * 0.05) &
            (df["Lower_wick"] < df["Body"] * 0.05)).astype(int)
        df["Bear_marubozu"] = ((df["Close"] < df["Open"]) &
            (df["Upper_wick"] < df["Body"] * 0.05) &
            (df["Lower_wick"] < df["Body"] * 0.05)).astype(int)
        df["Day_of_week"]    = df.index.dayofweek
        df["Week_of_month"]  = (df.index.day - 1) // 7 + 1
        df["Is_monday"]      = (df.index.dayofweek == 0).astype(int)
        df["Is_friday"]      = (df.index.dayofweek == 4).astype(int)
        df["Is_nfp_week"]    = (df["Week_of_month"] == 1).astype(int)
        df["Month"]          = df.index.month
        df["Is_end_month"]   = (df.index.day >= 25).astype(int)
        df["Is_quarter_end"] = (df.index.month.isin([3, 6, 9, 12]) &
            (df.index.day >= 25)).astype(int)
        df["Wyckoff_range_high"] = df["High"].rolling(50).max()
        df["Wyckoff_range_low"]  = df["Low"].rolling(50).min()
        df["Wyckoff_range"]      = df["Wyckoff_range_high"] - df["Wyckoff_range_low"]
        df["Wyckoff_accum"]      = ((df["Close"] < df["Wyckoff_range_low"] +
            df["Wyckoff_range"] * 0.3) &
            (df["Volume"] > df["Volume"].rolling(20).mean())).astype(int)
        df["Wyckoff_distrib"]    = ((df["Close"] > df["Wyckoff_range_high"] -
            df["Wyckoff_range"] * 0.3) &
            (df["Volume"] > df["Volume"].rolling(20).mean())).astype(int)
        df["Inst_momentum"]      = (df["Close"].rolling(20).mean() -
            df["Close"].rolling(50).mean())
        df["Inst_bull"]          = (df["Inst_momentum"] > 0).astype(int)
        df["Inst_bear"]          = (df["Inst_momentum"] < 0).astype(int)
        df["Vol_price_div_bull"] = ((df["Close"] < df["Close"].shift(5)) &
            (df["Volume"] > df["Volume"].rolling(20).mean() * 1.5)).astype(int)
        df["Vol_price_div_bear"] = ((df["Close"] > df["Close"].shift(5)) &
            (df["Volume"] > df["Volume"].rolling(20).mean() * 1.5)).astype(int)
        df["MSS_bull"]        = ((df["BOS_bull"] == 1) &
            (df["CHoCH_bull"] == 1)).astype(int)
        df["MSS_bear"]        = ((df["BOS_bear"] == 1) &
            (df["CHoCH_bear"] == 1)).astype(int)
        df["Inducement_bull"] = ((df["Equal_lows"] == 1) &
            (df["In_discount"] == 1)).astype(int)
        df["Inducement_bear"] = ((df["Equal_highs"] == 1) &
            (df["In_premium"] == 1)).astype(int)
        df["Global_bull_score"] = (df["SMC_bull_score"] + df["Inst_bull"] +
            df["OBV_trend"] + df["Above_VWAP"] + df["Above_PP"] +
            df["Ichi_bull"] + df["Wyckoff_accum"] + df["Bull_engulf"] +
            df["Hammer"] + df["Trend_bull_3TF"])
        df["Global_bear_score"] = (df["SMC_bear_score"] + df["Inst_bear"] +
            (1 - df["OBV_trend"]) + (1 - df["Above_VWAP"]) +
            (1 - df["Above_PP"]) + (1 - df["Ichi_bull"]) +
            df["Wyckoff_distrib"] + df["Bear_engulf"] +
            df["Shooting_star"] + df["Trend_bear_3TF"])
        df["Global_net_score"] = df["Global_bull_score"] - df["Global_bear_score"]
        for col in ["Sentiment_NLP", "Sentiment_bull", "Sentiment_bear",
                    "News_dans_30min", "News_dans_60min", "Nb_events",
                    "Surprise_bull", "Surprise_bear", "Macro_score_cal",
                    "NFP_change", "CPI_change", "GDP_change", "FED_RATE_change",
                    "PPI_change", "UNEMP_change", "REAL_RATE_change",
                    "BREAKEVEN_change", "Macro_bull_gold", "Macro_bear_gold",
                    "Macro_net_score"]:
            df[col] = 0.0
        for f in _ft:
            if f not in df.columns:
                df[f] = 0.0
        return df.ffill().dropna()
    except Exception as e:
        st.error("Erreur features: " + str(e))
        import traceback
        st.code(traceback.format_exc())
        return None

@st.cache_data(ttl=1800)
def get_sentiment():
    try:
        az = SentimentIntensityAnalyzer()
        urls = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
            "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en"
        ]
        titres = []
        for url in urls:
            try:
                r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.content, "lxml-xml")
                for item in soup.find_all("title")[1:8]:
                    if item.text.strip():
                        titres.append(item.text.strip())
            except:
                pass
        if not titres:
            return 0.0, []
        scores = [az.polarity_scores(t)["compound"] for t in titres]
        return sum(scores) / len(scores), titres
    except:
        return 0.0, []

@st.cache_data(ttl=900)
def get_calendar():
    events = []
    n30 = False
    n60 = False
    try:
        for url in [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
        ]:
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 100:
                    now = datetime.now(timezone.utc)
                    for ev in r.json():
                        if ev.get("impact") not in ["High", "Medium"]:
                            continue
                        try:
                            t = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                            d = (t - now).total_seconds() / 60
                            events.append({
                                "title": ev.get("title", ""),
                                "impact": ev.get("impact", ""),
                                "diff": d
                            })
                            if 0 < d <= 30:
                                n30 = True
                            if 0 < d <= 60:
                                n60 = True
                        except:
                            continue
                    break
            except:
                continue
    except:
        pass
    return events, n30, n60

@st.cache_data(ttl=3600)
def get_fred(key):
    try:
        fred = Fred(api_key=key)
        srs = {
            "NFP": "PAYEMS", "CPI": "CPIAUCSL", "GDP": "GDP",
            "FED_RATE": "FEDFUNDS", "PPI": "PPIACO", "UNEMP": "UNRATE",
            "REAL_RATE": "REAINTRATREARAT10Y", "BREAKEVEN": "T10YIE"
        }
        macro = {}
        for n, s in srs.items():
            try:
                d = fred.get_series(s, observation_start="2023-01-01")
                d.index = pd.to_datetime(d.index).tz_localize("UTC")
                macro[n] = d
            except:
                pass
        if not macro:
            return None
        m = pd.DataFrame(macro).ffill()
        m["NFP_change"]       = m["NFP"].diff()
        m["CPI_change"]       = m["CPI"].pct_change() * 100
        m["GDP_change"]       = m["GDP"].pct_change() * 100
        m["FED_RATE_change"]  = m["FED_RATE"].diff()
        m["PPI_change"]       = m["PPI"].pct_change() * 100
        m["UNEMP_change"]     = m["UNEMP"].diff()
        m["REAL_RATE_change"] = m["REAL_RATE"].diff()
        m["BREAKEVEN_change"] = m["BREAKEVEN"].diff()
        m["Macro_bull_gold"]  = (
            (m["CPI_change"] > 0).astype(int) +
            (m["REAL_RATE_change"] < 0).astype(int) +
            (m["UNEMP_change"] > 0).astype(int) +
            (m["NFP_change"] < 0).astype(int) +
            (m["BREAKEVEN_change"] > 0).astype(int))
        m["Macro_bear_gold"]  = (
            (m["CPI_change"] < 0).astype(int) +
            (m["REAL_RATE_change"] > 0).astype(int) +
            (m["UNEMP_change"] < 0).astype(int) +
            (m["NFP_change"] > 0).astype(int) +
            (m["BREAKEVEN_change"] < 0).astype(int))
        m["Macro_net_score"]  = m["Macro_bull_gold"] - m["Macro_bear_gold"]
        return m.dropna()
    except:
        return None

@st.cache_data(ttl=300)
def get_m15(signal, prix, atr_h1):
    try:
        m = download_data("GC=F", "15m", "5d")
        if m is None or len(m) < 50:
            return None
        m["RSI"]   = ta.momentum.rsi(m["Close"], 14)
        m["EMA20"] = ta.trend.ema_indicator(m["Close"], 20)
        m["EMA50"] = ta.trend.ema_indicator(m["Close"], 50)
        mc = ta.trend.MACD(m["Close"])
        m["MACD"]  = mc.macd()
        m["MACDS"] = mc.macd_signal()
        m["ATR"]   = ta.volatility.average_true_range(
            m["High"], m["Low"], m["Close"], 14)
        m["Trend"] = (m["EMA20"] > m["EMA50"]).astype(int)
        bb = ta.volatility.BollingerBands(m["Close"])
        m["BBpct"] = ((m["Close"] - bb.bollinger_lband()) /
                      (bb.bollinger_hband() - bb.bollinger_lband()))
        m["BOS_b"] = (m["High"] > m["High"].shift(1).rolling(5).max()).astype(int)
        m["BOS_s"] = (m["Low"] < m["Low"].shift(1).rolling(5).min()).astype(int)
        m["OB_b"]  = ((m["Close"].shift(1) < m["Open"].shift(1)) &
            (m["Close"] > m["Open"]) &
            (m["Close"] > m["High"].shift(1))).astype(int)
        m["OB_s"]  = ((m["Close"].shift(1) > m["Open"].shift(1)) &
            (m["Close"] < m["Open"]) &
            (m["Close"] < m["Low"].shift(1))).astype(int)
        m["FVG_b"] = ((m["Low"] > m["High"].shift(2)) &
            (m["Low"].shift(1) > m["High"].shift(2))).astype(int)
        m["FVG_s"] = ((m["High"] < m["Low"].shift(2)) &
            (m["High"].shift(1) < m["Low"].shift(2))).astype(int)
        m["HMR"]   = ((abs(m["Close"] - m["Open"]) * 2 
            (m[["Close", "Open"]].min(axis=1) - m["Low"])) &
            (m["Close"] > m["Open"])).astype(int)
        m["PIN"]   = ((abs(m["Close"] - m["Open"]) * 2 
            (m["High"] - m[["Close", "Open"]].max(axis=1))) &
            (m["Close"] < m["Open"])).astype(int)
        m["ENG_b"] = ((m["Close"].shift(1) < m["Open"].shift(1)) &
            (m["Close"] > m["Open"]) &
            (m["Close"] > m["Open"].shift(1)) &
            (m["Open"] < m["Close"].shift(1))).astype(int)
        m["ENG_s"] = ((m["Close"].shift(1) > m["Open"].shift(1)) &
            (m["Close"] < m["Open"]) &
            (m["Close"] < m["Open"].shift(1)) &
            (m["Open"] > m["Close"].shift(1))).astype(int)
        m = m.dropna()
        if len(m) < 20:
            return None
        d = m.iloc[-1]
        rsi = float(d["RSI"])
        macd = float(d["MACD"])
        macds = float(d["MACDS"])
        tr = int(d["Trend"])
        atr = float(d["ATR"])
        bb = float(d["BBpct"])
        p = float(d["Close"])
        sc = 0
        rs = []
        if signal == "LONG":
            if rsi < 50:
                sc += 2
                rs.append("RSI neutre (" + str(round(rsi, 1)) + ") ok")
            if rsi < 40:
                sc += 1
                rs.append("RSI survendu (" + str(round(rsi, 1)) + ") ok")
            if macd > macds:
                sc += 2
                rs.append("MACD haussier ok")
            if int(d["BOS_b"]) == 1:
                sc += 2
                rs.append("BOS haussier ok")
            if int(d["OB_b"]) == 1:
                sc += 3
                rs.append("Order Block haussier ok")
            if int(d["FVG_b"]) == 1:
                sc += 2
                rs.append("FVG haussier ok")
            if int(d["HMR"]) == 1:
                sc += 2
                rs.append("Hammer ok")
            if int(d["ENG_b"]) == 1:
                sc += 3
                rs.append("Engulfing haussier ok")
            if bb < 0.3:
                sc += 1
                rs.append("Prix bas BB ok")
            if tr == 1:
                sc += 1
                rs.append("Tendance M15 haussiere ok")
            sl = m["Low"].tail(10).min() - atr * 0.5
            tp1 = p + atr_h1 * 1.5
            tp2 = p + atr_h1 * 3.0
            el = p - atr * 0.3
            eh = p + atr * 0.2
        elif signal == "SHORT":
            if rsi > 50:
                sc += 2
                rs.append("RSI neutre (" + str(round(rsi, 1)) + ") ok")
            if rsi > 60:
                sc += 1
                rs.append("RSI suracheté (" + str(round(rsi, 1)) + ") ok")
            if macd < macds:
                sc += 2
                rs.append("MACD baissier ok")
            if int(d["BOS_s"]) == 1:
                sc += 2
                rs.append("BOS baissier ok")
            if int(d["OB_s"]) == 1:
                sc += 3
                rs.append("Order Block baissier ok")
            if int(d["FVG_s"]) == 1:
                sc += 2
                rs.append("FVG baissier ok")
            if int(d["PIN"]) == 1:
                sc += 2
                rs.append("Pin Bar baissiere ok")
            if int(d["ENG_s"]) == 1:
                sc += 3
                rs.append("Engulfing baissier ok")
            if bb > 0.7:
                sc += 1
                rs.append("Prix haut BB ok")
            if tr == 0:
                sc += 1
                rs.append("Tendance M15 baissiere ok")
            sl = m["High"].tail(10).max() + atr * 0.5
            tp1 = p - atr_h1 * 1.5
            tp2 = p - atr_h1 * 3.0
            el = p - atr * 0.2
            eh = p + atr * 0.3
        else:
            return None
        pct = sc / 17 * 100
        if pct >= 70:
            q, c, r = "A+", "#00ff88", "EXCELLENT - Entree recommandee"
        elif pct >= 50:
            q, c, r = "A", "#58a6ff", "BON SETUP - Entree possible"
        elif pct >= 35:
            q, c, r = "B", "#ffaa00", "MOYEN - Attendre confirmation"
        else:
            q, c, r = "C", "#ff4466", "FAIBLE - Ne pas trader"
        return {
            "q": q, "c": c, "r": r, "pct": pct, "rs": rs,
            "rsi": rsi, "atr": atr, "el": el, "eh": eh,
            "ep": p, "sl": sl, "tp1": tp1, "tp2": tp2, "df": m
        }
    except:
        return None

st.title("XAUUSD ML Trading Scanner")
st.caption("SMC + Technique + Macro + Sentiment | 62.63% WF | 66.95% HC")

mok, mmsg = is_market_open()
if not mok:
    st.warning("Marche ferme : " + mmsg)

with st.sidebar:
    st.header("Parametres")
    capital  = st.number_input("Capital (EUR)", 100, 100000, 500, 100)
    seuil    = st.slider("Seuil confiance", 0.60, 0.90, 0.65, 0.05)
    fred_key = st.text_input("Cle FRED API", "", type="password")
    if st.button("Rafraichir", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.metric("Walk-Forward", "62.63%")
    st.metric("Haute Confiance", "66.95%")
    st.metric("Fenetres", "7/7 > 55%")

mx, ml, mx2, ft, cfg = load_models()
if mx is None:
    st.stop()

with st.spinner("Chargement donnees..."):
    h1, d1 = get_all_data()

if not h1 or "GOLD" not in h1 or d1 is None:
    st.error("Impossible de charger Gold")
    st.stop()

with st.spinner("Calcul features..."):
    df = build_features(h1, d1, ft)

if df is None or len(df) == 0:
    st.error("Donnees insuffisantes")
    st.stop()

if fred_key:
    fd = get_fred(fred_key)
    if fd is not None:
        lr = fd.iloc[-1]
        for col in fd.columns:
            if col in df.columns:
                df[col] = float(lr[col])

sc, nt = get_sentiment()
df["Sentiment_NLP"]  = float(sc)
df["Sentiment_bull"] = int(sc > 0.05)
df["Sentiment_bear"] = int(sc < -0.05)

ev, n30, n60 = get_calendar()
df["News_dans_30min"] = int(n30)
df["News_dans_60min"] = int(n60)

row  = df.iloc[-1]
prix = float(row["Close"])
atr  = float(row["ATR"])
ts   = df.index[-1]

Xn = df[ft].iloc[[-1]].fillna(0)
p1 = mx.predict_proba(Xn)[0][1]
p2 = ml.predict_proba(Xn)[0][1]
p3 = mx2.predict_proba(Xn)[0][1]
pr = (p1 + p2 + p3) / 3
pl = pr
ps = 1 - pr

if pl > seuil:
    sig = "LONG"
    col = "#00ff88"
    em  = "📈"
elif ps > seuil:
    sig = "SHORT"
    col = "#ff4466"
    em  = "📉"
else:
    sig = "ATTENDRE"
    col = "#ffaa00"
    em  = "⏸️"

pos = get_position(max(pl, ps), capital, atr, prix, cfg.get("kelly_frac", 0.126))

c1, c2, c3 = st.columns([1, 2, 1])

with c1:
    ret = float(row.get("Return_1h", 0)) * 100
    st.metric("Prix XAUUSD", str(round(prix, 2)), str(round(ret, 2)) + "%")
    st.metric("Heure UTC", ts.strftime("%H:%M"))
    st.metric("Date", ts.strftime("%Y-%m-%d"))
    if n30:
        st.error("NEWS < 30min!")
    elif n60:
        st.warning("NEWS < 60min")
    else:
        st.success("Pas de news imminente")

with c2:
    score_val = round(max(pl, ps) * 100, 1)
    st.markdown(
        "<div style='background:#161b22;border:3px solid " + col +
        ";border-radius:15px;padding:30px;text-align:center;'>" +
        "<h1 style='color:" + col + ";font-size:52px;margin:0;'>" +
        em + " " + sig + "</h1>" +
        "<h2 style='color:" + col + ";margin:5px;'>Score : " +
        str(score_val) + "%</h2>" +
        "<p style='color:#888;'>XGB:" + str(round(p1*100,1)) +
        "% | LGB:" + str(round(p2*100,1)) +
        "% | XGB2:" + str(round(p3*100,1)) + "%</p>" +
        "</div>",
        unsafe_allow_html=True
    )

with c3:
    st.metric("LONG",  str(round(pl * 100, 1)) + "%")
    st.metric("SHORT", str(round(ps * 100, 1)) + "%")
    st.metric("ATR",   str(round(atr, 2)))

st.divider()

c4, c5, c6 = st.columns(3)

with c4:
    st.subheader("SL/TP Automatique")
    st.markdown(
        "**Capital :** " + str(capital) + " EUR\n\n" +
        "**Risque :** " + str(round(pos["risk_pct"], 2)) + "% = **" +
        str(round(pos["risk_eur"], 2)) + " EUR**\n\n" +
        "**ATR :** " + str(round(atr, 2)) + "\n\n" +
        "📈 LONG — SL: `" + str(round(pos["sl_long"], 2)) +
        "` TP: `" + str(round(pos["tp_long"], 2)) + "`\n\n" +
        "📉 SHORT — SL: `" + str(round(pos["sl_short"], 2)) +
        "` TP: `" + str(round(pos["tp_short"], 2)) + "`\n\n" +
        "**R/R :** 1:2.0"
    )

with c5:
    st.subheader("Indicateurs")
    rsi_v  = float(row.get("RSI", 50))
    rsi_h4 = float(row.get("RSI_H4", 50))
    rsi_d1 = float(row.get("RSI_D1", 50))
    th4    = int(row.get("Trend_H4", 0))
    td1    = int(row.get("Trend_D1", 0))
    smc_b  = int(row.get("SMC_bull_score", 0))
    smc_s  = int(row.get("SMC_bear_score", 0))
    gns    = int(row.get("Global_net_score", 0))
    st.metric("RSI H1", str(round(rsi_v, 1)),
              "Surachat" if rsi_v > 70 else ("Survente" if rsi_v < 30 else "Neutre"))
    st.metric("RSI H4", str(round(rsi_h4, 1)))
    st.metric("RSI D1", str(round(rsi_d1, 1)))
    st.metric("H4", "Haussier" if th4 else "Baissier")
    st.metric("D1", "Haussier" if td1 else "Baissier")
    st.metric("SMC", str(smc_b) + "/" + str(smc_s))
    st.metric("Score Global", str(gns))

with c6:
    st.subheader("Sentiment")
    sl2 = "POSITIF" if sc > 0.05 else ("NEGATIF" if sc < -0.05 else "NEUTRE")
    st.metric("Sentiment Gold", str(round(sc, 3)), sl2)
    az = SentimentIntensityAnalyzer()
    for t in nt[:4]:
        s = az.polarity_scores(t)["compound"]
        e = "ok" if s > 0.05 else ("nok" if s < -0.05 else "---")
        st.markdown(e + " " + t[:50])
    if fred_key and "Macro_net_score" in df.columns:
        st.divider()
        st.metric("Macro Net", str(int(row.get("Macro_net_score", 0))))

st.divider()
st.subheader("Calendrier Economique")
if ev:
    cc = st.columns(3)
    shown = 0
    for i, e in enumerate(ev):
        d = e["diff"]
        if d < -120:
            continue
        em2 = "HIGH" if e["impact"] == "High" else "MED"
        if -120 <= d <= 0:
            tm = "Il y a " + str(int(abs(d))) + "min"
        else:
            h2 = int(d // 60)
            m2 = int(d % 60)
            tm = "Dans " + str(h2) + "h" + str(m2).zfill(2) + "m"
        with cc[shown % 3]:
            st.markdown("**" + e["title"][:28] + "**")
            st.caption(em2 + " | " + tm)
        shown += 1
        if shown >= 9:
            break
else:
    st.info("Pas d evenements majeurs")

st.divider()
st.subheader("Analyse M15 - Entree Precise")
if sig != "ATTENDRE":
    with st.spinner("Analyse M15..."):
        s15 = get_m15(sig, prix, atr)
    if s15:
        ms1, ms2, ms3 = st.columns([1, 2, 1])
        with ms1:
            st.markdown(
                "<div style='background:#161b22;border:2px solid " + s15["c"] +
                ";border-radius:10px;padding:20px;text-align:center;'>" +
                "<h1 style='color:" + s15["c"] + ";font-size:56px;margin:0;'>" +
                s15["q"] + "</h1>" +
                "<p style='color:" + s15["c"] + ";'>Qualite Setup</p>" +
                "<p style='color:#888;'>Score: " + str(round(s15["pct"], 0)) + "%</p>" +
                "</div>",
                unsafe_allow_html=True
            )
        with ms2:
            st.markdown(
                "<div style='background:#161b22;border:1px solid #30363d;" +
                "border-radius:10px;padding:20px;'>" +
                "<h3 style='color:white;'>Zone Entree M15</h3>" +
                "<p style='color:#58a6ff;font-size:18px;'>Ideal : <b>" +
                str(round(s15["ep"], 2)) + "</b></p>" +
                "<p style='color:#888;'>Zone : " + str(round(s15["el"], 2)) +
                " -- " + str(round(s15["eh"], 2)) + "</p>" +
                "<hr style='border-color:#30363d;'>" +
                "<p style='color:#ff4466;'>SL : <b>" + str(round(s15["sl"], 2)) + "</b></p>" +
                "<p style='color:#00ff88;'>TP1 : <b>" + str(round(s15["tp1"], 2)) +
                "</b> (1:1.5)</p>" +
                "<p style='color:#00ff88;'>TP2 : <b>" + str(round(s15["tp2"], 2)) +
                "</b> (1:3)</p>" +
                "</div>",
                unsafe_allow_html=True
            )
        with ms3:
            st.metric("RSI M15", str(round(s15["rsi"], 1)))
            st.metric("ATR M15", str(round(s15["atr"], 2)))
        st.markdown(
            "<div style='background:#161b22;border:2px solid " + s15["c"] +
            ";border-radius:10px;padding:15px;text-align:center;'>" +
            "<h3 style='color:" + s15["c"] + ";'>" + s15["r"] + "</h3>" +
            "</div>",
            unsafe_allow_html=True
        )
        if s15["rs"]:
            st.markdown("**Confirmations M15 :**")
            cr = st.columns(2)
            for i, r2 in enumerate(s15["rs"]):
                with cr[i % 2]:
                    st.markdown("- " + r2)
        dm = s15["df"].tail(50)
        fm = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fm.add_trace(go.Candlestick(
            x=dm.index, open=dm["Open"], high=dm["High"],
            low=dm["Low"], close=dm["Close"],
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4466",
            name="M15"), row=1, col=1)
        fm.add_trace(go.Scatter(x=dm.index, y=dm["EMA20"],
            line=dict(color="#58a6ff", width=1), name="EMA20"), row=1, col=1)
        fm.add_trace(go.Scatter(x=dm.index, y=dm["EMA50"],
            line=dict(color="#ff8800", width=1), name="EMA50"), row=1, col=1)
        fm.add_hrect(y0=s15["el"], y1=s15["eh"],
            fillcolor="#58a6ff", opacity=0.1, row=1, col=1)
        fm.add_hline(y=s15["sl"], line_dash="dash",
            line_color="#ff4466", row=1, col=1)
        fm.add_hline(y=s15["tp1"], line_dash="dash",
            line_color="#00ff88", row=1, col=1)
        fm.add_hline(y=s15["tp2"], line_dash="dot",
            line_color="#00ff88", row=1, col=1)
        fm.add_trace(go.Scatter(x=dm.index, y=dm["RSI"],
            line=dict(color="#c678dd", width=1.5), name="RSI"), row=2, col=1)
        fm.add_hline(y=70, line_dash="dash", line_color="#ff4466", row=2, col=1)
        fm.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=2, col=1)
        fm.update_layout(height=500, paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22", font=dict(color="white"),
            xaxis_rangeslider_visible=False)
        fm.update_xaxes(gridcolor="#1e2d3d")
        fm.update_yaxes(gridcolor="#1e2d3d")
        st.plotly_chart(fm, use_container_width=True)
    else:
        st.info("Donnees M15 insuffisantes")
else:
    st.info("Signal ATTENDRE - Pas de setup M15 a analyser")

st.divider()
st.subheader("XAUUSD H1 - 100 dernieres bougies")
dc = df.tail(100)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2])
fig.add_trace(go.Candlestick(
    x=dc.index, open=dc["Open"], high=dc["High"],
    low=dc["Low"], close=dc["Close"],
    increasing_line_color="#00ff88",
    decreasing_line_color="#ff4466",
    name="XAUUSD"), row=1, col=1)
for ema, ce, ne in [
    ("EMA20", "#58a6ff", "EMA20"),
    ("EMA50", "#ff8800", "EMA50"),
    ("EMA200", "#ff4466", "EMA200")
]:
    if ema in dc.columns:
        fig.add_trace(go.Scatter(x=dc.index, y=dc[ema],
            line=dict(color=ce, width=1), name=ne), row=1, col=1)
if "RSI" in dc.columns:
    fig.add_trace(go.Scatter(x=dc.index, y=dc["RSI"],
        line=dict(color="#c678dd", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4466", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=2, col=1)
if "MACD" in dc.columns:
    fig.add_trace(go.Scatter(x=dc.index, y=dc["MACD"],
        line=dict(color="#00ff88", width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dc.index, y=dc["MACD_signal"],
        line=dict(color="#ff4466", width=1.5), name="Signal"), row=3, col=1)
    colors_hist = ["#00ff88" if v >= 0 else "#ff4466" for v in dc["MACD_hist"]]
    fig.add_bar(x=dc.index, y=dc["MACD_hist"],
        marker_color=colors_hist, name="Hist", row=3, col=1)
fig.update_layout(height=700, paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22", font=dict(color="white"),
    xaxis_rangeslider_visible=False, showlegend=True)
fig.update_xaxes(gridcolor="#1e2d3d")
fig.update_yaxes(gridcolor="#1e2d3d")
st.plotly_chart(fig, use_container_width=True)

now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
st.caption("Pas un conseil financier | MAJ: " + now_str)
