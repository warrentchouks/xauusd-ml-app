
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

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="XAUUSD ML Scanner",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style dark
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }
    h1, h2, h3 { color: #ffffff; }
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        border: 1px solid #30363d;
    }
    .signal-long {
        background-color: #0d2818;
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .signal-short {
        background-color: #2d0f0f;
        border: 2px solid #ff4466;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .signal-neutral {
        background-color: #1a1a0f;
        border: 2px solid #ffaa00;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT MODÈLES
# ============================================
@st.cache_resource
def load_models():
    try:
        model_xgb  = joblib.load("model_xgb_final.pkl")
        model_lgb  = joblib.load("model_lgb_final.pkl")
        model_xgb2 = joblib.load("model_xgb2_final.pkl")
        features   = joblib.load("features_final.pkl")
        with open("config.json") as f:
            config = json.load(f)
        return model_xgb, model_lgb, model_xgb2, features, config
    except Exception as e:
        st.error(f"Erreur chargement modèles: {e}")
        return None, None, None, None, None

model_xgb, model_lgb, model_xgb2, features, config = load_models()

# ============================================
# FONCTIONS DONNÉES
# ============================================
@st.cache_data(ttl=900)  # Cache 15 minutes
def get_market_data():
    actifs = {
        "GOLD"  : "GC=F",
        "DXY"   : "DX-Y.NYB",
        "VIX"   : "^VIX",
        "US10Y" : "^TNX",
        "SILVER": "SI=F",
        "SP500" : "^GSPC",
    }
    data = {}
    for nom, ticker in actifs.items():
        try:
            df = yf.download(ticker, interval="1h", period="60d")
            # Correction multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c) for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            # Garder seulement colonnes OHLCV
            cols = [c for c in df.columns
                   if c in ["Open","High","Low","Close","Volume"]]
            df = df[cols]
            if len(df) > 0:
                data[nom] = df
        except Exception as e:
            pass
    return data

@st.cache_data(ttl=900)
def prepare_features(data, features, config):
    try:
        gold = data["GOLD"].copy()

        def aligner(data_dict, nom):
            df = data_dict[nom].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["Close"]].rename(columns={"Close": nom})

        df = gold.copy()
        for nom in ["DXY","VIX","US10Y","SILVER","SP500"]:
            if nom in data:
                df = pd.merge_asof(
                    df.sort_index(),
                    aligner(data, nom).sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward")

        df = df.ffill().dropna()

        # H4 et D1
        def resample(df, tf):
            return df[["Open","High","Low","Close","Volume"]].resample(tf).agg({
                "Open":"first","High":"max",
                "Low":"min","Close":"last","Volume":"sum"
            }).dropna()

        df_h4 = resample(df, "4h")
        df_d1 = resample(df, "1D")

        df_h4["EMA20_H4"] = ta.trend.ema_indicator(df_h4["Close"], 20)
        df_h4["EMA50_H4"] = ta.trend.ema_indicator(df_h4["Close"], 50)
        df_h4["RSI_H4"]   = ta.momentum.rsi(df_h4["Close"], 14)
        df_h4["Trend_H4"] = (df_h4["EMA20_H4"] > df_h4["EMA50_H4"]).astype(int)

        df_d1["EMA20_D1"]  = ta.trend.ema_indicator(df_d1["Close"], 20)
        df_d1["EMA50_D1"]  = ta.trend.ema_indicator(df_d1["Close"], 50)
        df_d1["EMA200_D1"] = ta.trend.ema_indicator(df_d1["Close"], 200)
        df_d1["RSI_D1"]    = ta.momentum.rsi(df_d1["Close"], 14)
        df_d1["Trend_D1"]  = (df_d1["EMA20_D1"] > df_d1["EMA50_D1"]).astype(int)

        df = pd.merge_asof(df.sort_index(),
            df_h4[["Trend_H4","EMA20_H4","EMA50_H4","RSI_H4"]].sort_index(),
            left_index=True, right_index=True, direction="backward")
        df = pd.merge_asof(df.sort_index(),
            df_d1[["Trend_D1","EMA20_D1","EMA50_D1","EMA200_D1","RSI_D1"]].sort_index(),
            left_index=True, right_index=True, direction="backward")

        # Features techniques
        df["RSI"]     = ta.momentum.rsi(df["Close"], 14)
        df["RSI_ob"]  = (df["RSI"] > 70).astype(int)
        df["RSI_os"]  = (df["RSI"] < 30).astype(int)
        df["EMA20"]   = ta.trend.ema_indicator(df["Close"], 20)
        df["EMA50"]   = ta.trend.ema_indicator(df["Close"], 50)
        df["EMA200"]  = ta.trend.ema_indicator(df["Close"], 200)

        macd = ta.trend.MACD(df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"]   = macd.macd_diff()
        df["MACD_cross_up"]   = ((df["MACD"] > df["MACD_signal"]) &
                                  (df["MACD"].shift(1) < df["MACD_signal"].shift(1))).astype(int)
        df["MACD_cross_down"] = ((df["MACD"] < df["MACD_signal"]) &
                                  (df["MACD"].shift(1) > df["MACD_signal"].shift(1))).astype(int)

        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = df["BB_upper"] - df["BB_lower"]
        df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / df["BB_width"]

        df["ATR"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], 14)

        df["Return_1h"]  = df["Close"].pct_change(1)
        df["Return_4h"]  = df["Close"].pct_change(4)
        df["Return_24h"] = df["Close"].pct_change(24)

        df["Price_vs_EMA20"]  = (df["Close"] - df["EMA20"])  / df["EMA20"]  * 100
        df["Price_vs_EMA50"]  = (df["Close"] - df["EMA50"])  / df["EMA50"]  * 100
        df["Price_vs_EMA200"] = (df["Close"] - df["EMA200"]) / df["EMA200"] * 100
        df["EMA20_vs_EMA50"]  = (df["EMA20"] - df["EMA50"])  / df["EMA50"]  * 100
        df["High_Low_pct"]    = (df["High"] - df["Low"]) / df["Close"] * 100

        df["Hour"]             = df.index.hour
        df["Session_London"]   = ((df["Hour"] >= 7)  & (df["Hour"] < 16)).astype(int)
        df["Session_NY"]       = ((df["Hour"] >= 13) & (df["Hour"] < 21)).astype(int)
        df["Session_Overlap"]  = ((df["Hour"] >= 13) & (df["Hour"] < 16)).astype(int)
        df["Session_Asian"]    = ((df["Hour"] >= 0)  & (df["Hour"] < 7)).astype(int)

        df["DXY_return"]    = df["DXY"].pct_change(4)
        df["VIX_return"]    = df["VIX"].pct_change(4)
        df["SILVER_return"] = df["SILVER"].pct_change(4)
        df["SP500_return"]  = df["SP500"].pct_change(4)
        df["Gold_DXY_div"]  = df["Return_4h"] + df["DXY_return"]

        df["Trend_bull_3TF"] = (
            (df["Trend_D1"] == 1) & (df["Trend_H4"] == 1)).astype(int)
        df["Trend_bear_3TF"] = (
            (df["Trend_D1"] == 0) & (df["Trend_H4"] == 0)).astype(int)

        # SMC
        df["BOS_bull"] = (df["High"] > df["High"].shift(1).rolling(10).max()).astype(int)
        df["BOS_bear"] = (df["Low"]  < df["Low"].shift(1).rolling(10).min()).astype(int)
        df["Higher_High"] = (df["High"] > df["High"].shift(1)).astype(int)
        df["Lower_Low"]   = (df["Low"]  < df["Low"].shift(1)).astype(int)
        df["CHoCH_bear"]  = ((df["Higher_High"].shift(3)==1) &
                              (df["Higher_High"].shift(2)==1) &
                              (df["Lower_Low"]==1)).astype(int)
        df["CHoCH_bull"]  = ((df["Lower_Low"].shift(3)==1) &
                              (df["Lower_Low"].shift(2)==1) &
                              (df["Higher_High"]==1)).astype(int)
        df["Bearish_candle"] = (df["Close"] < df["Open"]).astype(int)
        df["Bullish_candle"] = (df["Close"] > df["Open"]).astype(int)
        df["Strong_up"]   = (df["Return_1h"] >  df["ATR"]/df["Close"]*100).astype(int)
        df["Strong_down"] = (df["Return_1h"] < -df["ATR"]/df["Close"]*100).astype(int)
        df["OB_bull"] = ((df["Bearish_candle"].shift(1)==1) & (df["Strong_up"]==1)).astype(int)
        df["OB_bear"] = ((df["Bullish_candle"].shift(1)==1) & (df["Strong_down"]==1)).astype(int)
        df["Range_high"]  = df["High"].rolling(100).max()
        df["Range_low"]   = df["Low"].rolling(100).min()
        df["Range_mid"]   = (df["Range_high"] + df["Range_low"]) / 2
        df["In_discount"] = (df["Close"] < df["Range_mid"]).astype(int)
        df["In_premium"]  = (df["Close"] > df["Range_mid"]).astype(int)
        tolerance = 0.001
        df["Equal_highs"] = (abs(df["High"]-df["High"].shift(1))/df["High"] < tolerance).astype(int)
        df["Equal_lows"]  = (abs(df["Low"]-df["Low"].shift(1))/df["Low"]   < tolerance).astype(int)
        df["SMC_bull_score"] = (df["BOS_bull"]+df["CHoCH_bull"]+df["OB_bull"]+df["In_discount"])
        df["SMC_bear_score"] = (df["BOS_bear"]+df["CHoCH_bear"]+df["OB_bear"]+df["In_premium"])
        df["SMC_bull_confirm"] = ((df["SMC_bull_score"]>=2)&(df["Trend_bull_3TF"]==1)).astype(int)
        df["SMC_bear_confirm"] = ((df["SMC_bear_score"]>=2)&(df["Trend_bear_3TF"]==1)).astype(int)

        # Indicateurs avancés
        stoch = ta.momentum.StochasticOscillator(df["High"],df["Low"],df["Close"],14)
        df["Stoch_K"]          = stoch.stoch()
        df["Stoch_D"]          = stoch.stoch_signal()
        df["Stoch_ob"]         = (df["Stoch_K"] > 80).astype(int)
        df["Stoch_os"]         = (df["Stoch_K"] < 20).astype(int)
        df["Stoch_cross_up"]   = ((df["Stoch_K"]>df["Stoch_D"])&
                                   (df["Stoch_K"].shift(1)<df["Stoch_D"].shift(1))).astype(int)
        df["Stoch_cross_down"] = ((df["Stoch_K"]<df["Stoch_D"])&
                                   (df["Stoch_K"].shift(1)>df["Stoch_D"].shift(1))).astype(int)
        df["Williams_R"]  = ta.momentum.williams_r(df["High"],df["Low"],df["Close"],14)
        df["Williams_ob"] = (df["Williams_R"] > -20).astype(int)
        df["Williams_os"] = (df["Williams_R"] < -80).astype(int)
        df["CCI"]    = ta.trend.cci(df["High"],df["Low"],df["Close"],20)
        df["CCI_ob"] = (df["CCI"] >  100).astype(int)
        df["CCI_os"] = (df["CCI"] < -100).astype(int)

        ichi = ta.trend.IchimokuIndicator(df["High"],df["Low"],9,26,52)
        df["Ichi_tenkan"]   = ichi.ichimoku_conversion_line()
        df["Ichi_kijun"]    = ichi.ichimoku_base_line()
        df["Ichi_spanA"]    = ichi.ichimoku_a()
        df["Ichi_spanB"]    = ichi.ichimoku_b()
        df["Ichi_bull"]     = (df["Close"] > df["Ichi_spanA"]).astype(int)
        df["Ichi_tk_cross"] = (df["Ichi_tenkan"] > df["Ichi_kijun"]).astype(int)

        df["VWAP"]          = (df["Close"]*df["Volume"]).cumsum()/df["Volume"].cumsum()
        df["Price_vs_VWAP"] = (df["Close"]-df["VWAP"])/df["VWAP"]*100
        df["Above_VWAP"]    = (df["Close"] > df["VWAP"]).astype(int)

        df["OBV"]       = ta.volume.on_balance_volume(df["Close"],df["Volume"])
        df["OBV_EMA"]   = ta.trend.ema_indicator(df["OBV"],20)
        df["OBV_trend"] = (df["OBV"] > df["OBV_EMA"]).astype(int)
        df["OBV_divergence"] = ((df["Close"]>df["Close"].shift(5))&
                                 (df["OBV"]<df["OBV"].shift(5))).astype(int)

        # Pivot Points
        df_d1_pivot = resample(df, "1D")
        df_d1_pivot["PP"] = (df_d1_pivot["High"]+df_d1_pivot["Low"]+df_d1_pivot["Close"])/3
        df_d1_pivot["R1"] = 2*df_d1_pivot["PP"]-df_d1_pivot["Low"]
        df_d1_pivot["S1"] = 2*df_d1_pivot["PP"]-df_d1_pivot["High"]
        df_d1_pivot["R2"] = df_d1_pivot["PP"]+(df_d1_pivot["High"]-df_d1_pivot["Low"])
        df_d1_pivot["S2"] = df_d1_pivot["PP"]-(df_d1_pivot["High"]-df_d1_pivot["Low"])
        df_d1_pivot["R3"] = df_d1_pivot["High"]+2*(df_d1_pivot["PP"]-df_d1_pivot["Low"])
        df_d1_pivot["S3"] = df_d1_pivot["Low"]-2*(df_d1_pivot["High"]-df_d1_pivot["PP"])
        pivot_cols = df_d1_pivot[["PP","R1","S1","R2","S2","R3","S3"]].shift(1)
        df = pd.merge_asof(df.sort_index(),pivot_cols.sort_index(),
                           left_index=True,right_index=True,direction="backward")
        df["Dist_PP"]  = (df["Close"]-df["PP"])/df["Close"]*100
        df["Dist_R1"]  = (df["R1"]-df["Close"])/df["Close"]*100
        df["Dist_S1"]  = (df["Close"]-df["S1"])/df["Close"]*100
        df["Near_PP"]  = (abs(df["Dist_PP"])<0.1).astype(int)
        df["Near_R1"]  = (abs(df["Dist_R1"])<0.1).astype(int)
        df["Near_S1"]  = (abs(df["Dist_S1"])<0.1).astype(int)
        df["Above_PP"] = (df["Close"] > df["PP"]).astype(int)

        # Chandeliers
        df["Body"]          = abs(df["Close"]-df["Open"])
        df["Upper_wick"]    = df["High"]-df[["Close","Open"]].max(axis=1)
        df["Lower_wick"]    = df[["Close","Open"]].min(axis=1)-df["Low"]
        df["Range_bar"]     = df["High"]-df["Low"]
        df["Doji"]          = (df["Body"]<df["Range_bar"]*0.1).astype(int)
        df["Hammer"]        = ((df["Lower_wick"]>df["Body"]*2)&
                                (df["Upper_wick"]<df["Body"]*0.5)&
                                (df["Close"]>df["Open"])).astype(int)
        df["Shooting_star"] = ((df["Upper_wick"]>df["Body"]*2)&
                                (df["Lower_wick"]<df["Body"]*0.5)&
                                (df["Close"]<df["Open"])).astype(int)
        df["Bull_engulf"]   = ((df["Close"].shift(1)<df["Open"].shift(1))&
                                (df["Close"]>df["Open"])&
                                (df["Close"]>df["Open"].shift(1))&
                                (df["Open"]<df["Close"].shift(1))).astype(int)
        df["Bear_engulf"]   = ((df["Close"].shift(1)>df["Open"].shift(1))&
                                (df["Close"]<df["Open"])&
                                (df["Close"]<df["Open"].shift(1))&
                                (df["Open"]>df["Close"].shift(1))).astype(int)
        df["Bull_marubozu"] = ((df["Close"]>df["Open"])&
                                (df["Upper_wick"]<df["Body"]*0.05)&
                                (df["Lower_wick"]<df["Body"]*0.05)).astype(int)
        df["Bear_marubozu"] = ((df["Close"]<df["Open"])&
                                (df["Upper_wick"]<df["Body"]*0.05)&
                                (df["Lower_wick"]<df["Body"]*0.05)).astype(int)

        # Cycles
        df["Day_of_week"]    = df.index.dayofweek
        df["Week_of_month"]  = (df.index.day-1)//7+1
        df["Is_monday"]      = (df.index.dayofweek==0).astype(int)
        df["Is_friday"]      = (df.index.dayofweek==4).astype(int)
        df["Is_nfp_week"]    = (df["Week_of_month"]==1).astype(int)
        df["Month"]          = df.index.month
        df["Is_end_month"]   = (df.index.day>=25).astype(int)
        df["Is_quarter_end"] = (df.index.month.isin([3,6,9,12])&
                                 (df.index.day>=25)).astype(int)

        # FVG corrigé
        df["FVG_bull_clean"] = ((df["Low"]>df["High"].shift(2))&
                                  (df["Low"].shift(1)>df["High"].shift(2))).astype(int)
        df["FVG_bear_clean"] = ((df["High"]<df["Low"].shift(2))&
                                  (df["High"].shift(1)<df["Low"].shift(2))).astype(int)

        # Wyckoff
        df["Wyckoff_range_high"] = df["High"].rolling(50).max()
        df["Wyckoff_range_low"]  = df["Low"].rolling(50).min()
        df["Wyckoff_range"]      = df["Wyckoff_range_high"]-df["Wyckoff_range_low"]
        df["Wyckoff_accum"]      = ((df["Close"]<df["Wyckoff_range_low"]+df["Wyckoff_range"]*0.3)&
                                     (df["Volume"]>df["Volume"].rolling(20).mean())).astype(int)
        df["Wyckoff_distrib"]    = ((df["Close"]>df["Wyckoff_range_high"]-df["Wyckoff_range"]*0.3)&
                                     (df["Volume"]>df["Volume"].rolling(20).mean())).astype(int)

        # COT approximé
        df["Inst_momentum"]       = df["Close"].rolling(20).mean()-df["Close"].rolling(50).mean()
        df["Inst_bull"]           = (df["Inst_momentum"]>0).astype(int)
        df["Inst_bear"]           = (df["Inst_momentum"]<0).astype(int)
        df["Vol_price_div_bull"]  = ((df["Close"]<df["Close"].shift(5))&
                                      (df["Volume"]>df["Volume"].rolling(20).mean()*1.5)).astype(int)
        df["Vol_price_div_bear"]  = ((df["Close"]>df["Close"].shift(5))&
                                      (df["Volume"]>df["Volume"].rolling(20).mean()*1.5)).astype(int)

        # MSS + Inducement
        df["MSS_bull"]          = ((df["BOS_bull"]==1)&(df["CHoCH_bull"]==1)).astype(int)
        df["MSS_bear"]          = ((df["BOS_bear"]==1)&(df["CHoCH_bear"]==1)).astype(int)
        df["Inducement_bull"]   = ((df["Equal_lows"]==1)&(df["In_discount"]==1)).astype(int)
        df["Inducement_bear"]   = ((df["Equal_highs"]==1)&(df["In_premium"]==1)).astype(int)

        # Score global
        df["Global_bull_score"] = (df["SMC_bull_score"]+df["Inst_bull"]+
                                    df["OBV_trend"]+df["Above_VWAP"]+
                                    df["Above_PP"]+df["Ichi_bull"]+
                                    df["Wyckoff_accum"]+df["Bull_engulf"]+
                                    df["Hammer"]+df["Trend_bull_3TF"])
        df["Global_bear_score"] = (df["SMC_bear_score"]+df["Inst_bear"]+
                                    (1-df["OBV_trend"])+(1-df["Above_VWAP"])+
                                    (1-df["Above_PP"])+(1-df["Ichi_bull"])+
                                    df["Wyckoff_distrib"]+df["Bear_engulf"]+
                                    df["Shooting_star"]+df["Trend_bear_3TF"])
        df["Global_net_score"]  = df["Global_bull_score"]-df["Global_bear_score"]

        df = df.dropna()
        return df

    except Exception as e:
        st.error(f"Erreur préparation features: {e}")
        return None

def get_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    urls = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
    ]
    titres = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=5,
                               headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.content, "lxml-xml")
            for item in soup.find_all("title")[1:8]:
                if item.text.strip():
                    titres.append(item.text.strip())
        except:
            pass
    if not titres:
        return 0.0, []
    scores = [analyzer.polarity_scores(t)["compound"] for t in titres]
    return sum(scores)/len(scores), titres

def get_calendar():
    events = []
    news_30 = False
    news_60 = False
    try:
        for url in ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"]:
            try:
                resp = requests.get(url, timeout=8,
                                   headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200 and len(resp.content) > 100:
                    data = resp.json()
                    now  = datetime.now(timezone.utc)
                    for ev in data:
                        if ev.get("impact") not in ["High","Medium"]:
                            continue
                        try:
                            t    = datetime.fromisoformat(ev["date"].replace("Z","+00:00"))
                            diff = (t-now).total_seconds()/60
                            events.append({"title":ev.get("title",""),
                                          "impact":ev.get("impact",""),
                                          "diff":diff})
                            if 0 < diff <= 30: news_30 = True
                            if 0 < diff <= 60: news_60 = True
                        except:
                            continue
                    break
            except:
                continue
    except:
        pass
    return events, news_30, news_60

def calculer_position(proba, capital, atr, prix, kelly_frac, rr=2.0):
    if proba >= 0.85:
        risk_pct = min(kelly_frac*2.0, 0.03)
    elif proba >= 0.80:
        risk_pct = min(kelly_frac*1.5, 0.025)
    elif proba >= 0.75:
        risk_pct = min(kelly_frac*1.0, 0.02)
    elif proba >= 0.70:
        risk_pct = min(kelly_frac*0.75, 0.015)
    else:
        risk_pct = min(kelly_frac*0.5, 0.01)
    sl_dist  = 1.5*atr
    return {
        "risk_pct"   : risk_pct*100,
        "risk_amount": capital*risk_pct,
        "sl_long"    : prix-sl_dist,
        "tp_long"    : prix+sl_dist*rr,
        "sl_short"   : prix+sl_dist,
        "tp_short"   : prix-sl_dist*rr,
        "sl_dist"    : sl_dist
    }

# ============================================
# INTERFACE PRINCIPALE
# ============================================
st.title("🥇 XAUUSD ML Trading Scanner")
st.markdown("*Système ML multi-modèles — SMC + Technique + Macro + Sentiment*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    capital    = st.number_input("Capital (€)", 100, 100000, 500, 100)
    seuil_conf = st.slider("Seuil confiance", 0.60, 0.90, 0.65, 0.05)
    auto_refresh = st.checkbox("Auto-refresh (15min)", value=False)
    st.divider()
    st.markdown("**📊 Performances du modèle :**")
    st.metric("Walk-Forward Accuracy", "62.63%")
    st.metric("Haute Confiance", "66.95%")
    st.metric("Fenêtres rentables", "7/7")
    st.divider()
    if st.button("🔄 Rafraîchir les données", type="primary"):
        st.cache_data.clear()
        st.rerun()

# Chargement données
with st.spinner("📥 Chargement données marché..."):
    market_data = get_market_data()

if not market_data or "GOLD" not in market_data:
    st.error("❌ Impossible de charger les données")
    st.stop()

with st.spinner("🔧 Calcul des features..."):
    df_app = prepare_features(market_data, features, config)

if df_app is None or len(df_app) == 0:
    st.error("❌ Erreur dans le calcul des features")
    st.stop()

# Données actuelles
derniere    = df_app.iloc[-1]
prix_actuel = float(derniere["Close"])
atr_actuel  = float(derniere["ATR"])
timestamp   = df_app.index[-1]

# ============================================
# SCORING ML
# ============================================
exclude_cols = ["Open","High","Low","Close","Volume",
                "DXY","VIX","US10Y","SILVER","SP500",
                "Range_high","Range_low","Range_mid",
                "Higher_High","Lower_Low",
                "Bearish_candle","Bullish_candle",
                "Strong_up","Strong_down",
                "Wyckoff_range_high","Wyckoff_range_low",
                "PP","R1","S1","R2","S2","R3","S3",
                "Body","Upper_wick","Lower_wick","Range_bar",
                "VWAP","Inst_momentum"]

feat_disponibles = [f for f in features
                    if f in df_app.columns
                    and f not in exclude_cols]

X_now = df_app[feat_disponibles].iloc[[-1]]

try:
    p_xgb  = model_xgb.predict_proba(X_now)[0][1]
    p_lgb  = model_lgb.predict_proba(X_now)[0][1]
    p_xgb2 = model_xgb2.predict_proba(X_now)[0][1]
    proba  = (p_xgb + p_lgb + p_xgb2) / 3
except Exception as e:
    st.error(f"Erreur scoring: {e}")
    proba = 0.5

proba_long  = proba
proba_short = 1 - proba

# Signal
if proba_long > seuil_conf:
    signal = "LONG"
    signal_color = "#00ff88"
    signal_emoji = "📈"
elif proba_short > seuil_conf:
    signal = "SHORT"
    signal_color = "#ff4466"
    signal_emoji = "📉"
else:
    signal = "ATTENDRE"
    signal_color = "#ffaa00"
    signal_emoji = "⏸️"

# Sentiment + Calendrier
sentiment_score, news_titres = get_sentiment()
events, news_30, news_60     = get_calendar()

# Position sizing
kelly_frac = config.get("kelly_frac", 0.126)
pos = calculer_position(max(proba_long, proba_short),
                        capital, atr_actuel, prix_actuel, kelly_frac)

# ============================================
# AFFICHAGE
# ============================================

# Ligne 1 — Signal principal
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.metric("💰 Prix XAUUSD", f"{prix_actuel:.2f}",
              f"{derniere['Return_1h']*100:.2f}%")
    st.metric("⏰ Timestamp", timestamp.strftime("%H:%M UTC"))

with col2:
    score_display = max(proba_long, proba_short) * 100
    st.markdown(f"""
    <div style="background-color:#161b22; border:2px solid {signal_color};
                border-radius:15px; padding:25px; text-align:center;">
        <h1 style="color:{signal_color}; font-size:48px; margin:0;">
            {signal_emoji} {signal}
        </h1>
        <h2 style="color:{signal_color}; margin:5px;">
            Score : {score_display:.1f}%
        </h2>
        <p style="color:#888; margin:0;">
            XGB: {p_xgb*100:.1f}% |
            LGB: {p_lgb*100:.1f}% |
            XGB2: {p_xgb2*100:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.metric("📈 LONG",  f"{proba_long*100:.1f}%")
    st.metric("📉 SHORT", f"{proba_short*100:.1f}%")

st.divider()

# Ligne 2 — SL/TP + Calendrier
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🎯 SL/TP Automatique")
    st.markdown(f"""
    **Capital :** {capital}€
    **Risque :** {pos["risk_pct"]:.2f}% = **{pos["risk_amount"]:.2f}€**
    **ATR :** {atr_actuel:.2f}

    📈 **LONG**
    - Entry : `{prix_actuel:.2f}`
    - SL : `{pos["sl_long"]:.2f}`
    - TP : `{pos["tp_long"]:.2f}`

    📉 **SHORT**
    - Entry : `{prix_actuel:.2f}`
    - SL : `{pos["sl_short"]:.2f}`
    - TP : `{pos["tp_short"]:.2f}`
    """)

with col5:
    st.subheader("📅 Calendrier Économique")
    if news_30:
        st.error("🚨 NEWS DANS 30 MIN — NE PAS TRADER")
    elif news_60:
        st.warning("⚠️ News dans 60 min — Prudence")
    else:
        st.success("✅ Pas de news imminente")

    for ev in events[:8]:
        diff = ev["diff"]
        emoji = "🔴" if ev["impact"]=="High" else "🟡"
        if diff > 0:
            h, m = int(diff//60), int(diff%60)
            timing = f"Dans {h}h{m:02d}m"
        else:
            timing = f"Il y a {abs(diff):.0f}min"
        st.markdown(f"{emoji} **{ev['title'][:30]}** — {timing}")

with col6:
    st.subheader("📰 Sentiment NLP")
    sent_color = "🟢" if sentiment_score > 0.05 else ("🔴" if sentiment_score < -0.05 else "⚪")
    sent_label = "POSITIF" if sentiment_score > 0.05 else ("NÉGATIF" if sentiment_score < -0.05 else "NEUTRE")
    st.metric("Score Sentiment", f"{sentiment_score:.3f}", sent_label)
    for titre in news_titres[:5]:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        a = SentimentIntensityAnalyzer()
        s = a.polarity_scores(titre)["compound"]
        e = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
        st.markdown(f"{e} {titre[:55]}")

st.divider()

# Ligne 3 — Indicateurs techniques
st.subheader("📊 Indicateurs Techniques")
col7, col8, col9, col10 = st.columns(4)

with col7:
    rsi_val = float(derniere["RSI"])
    rsi_d1  = float(derniere["RSI_D1"])
    rsi_h4  = float(derniere["RSI_H4"])
    st.metric("RSI H1",  f"{rsi_val:.1f}",
              "Surachat" if rsi_val>70 else ("Survente" if rsi_val<30 else "Neutre"))
    st.metric("RSI H4",  f"{rsi_h4:.1f}")
    st.metric("RSI D1",  f"{rsi_d1:.1f}")

with col8:
    macd_val  = float(derniere["MACD"])
    macd_sig  = float(derniere["MACD_signal"])
    macd_hist = float(derniere["MACD_hist"])
    st.metric("MACD",    f"{macd_val:.2f}",
              "🟢 Haussier" if macd_val>macd_sig else "🔴 Baissier")
    st.metric("Signal",  f"{macd_sig:.2f}")
    st.metric("Histogramme", f"{macd_hist:.2f}")

with col9:
    trend_h4 = int(derniere["Trend_H4"])
    trend_d1 = int(derniere["Trend_D1"])
    bull_3tf  = int(derniere["Trend_bull_3TF"])
    st.metric("Tendance H4", "🟢 Haussier" if trend_h4 else "🔴 Baissier")
    st.metric("Tendance D1", "🟢 Haussier" if trend_d1 else "🔴 Baissier")
    st.metric("3TF aligné",  "✅ OUI" if bull_3tf else "❌ NON")

with col10:
    smc_bull = int(derniere["SMC_bull_score"])
    smc_bear = int(derniere["SMC_bear_score"])
    global_net = int(derniere["Global_net_score"])
    st.metric("SMC Bull Score", f"{smc_bull}/5")
    st.metric("SMC Bear Score", f"{smc_bear}/5")
    st.metric("Global Score",   f"{global_net:+d}",
              "🟢 Bull" if global_net>0 else "🔴 Bear")

st.divider()

# Graphique prix
st.subheader("📈 Prix XAUUSD H1 — 100 dernières bougies")
df_chart = df_app.tail(100)

fig = make_subplots(rows=3, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=["Prix + EMA + Pivots",
                                   "RSI", "MACD"])

# Chandeliers
fig.add_trace(go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"], high=df_chart["High"],
    low=df_chart["Low"],   close=df_chart["Close"],
    name="XAUUSD",
    increasing_line_color="#00ff88",
    decreasing_line_color="#ff4466"), row=1, col=1)

# EMA
for ema, color, name in [
    ("EMA20","#58a6ff","EMA20"),
    ("EMA50","#ff8800","EMA50"),
    ("EMA200","#ff4466","EMA200")]:
    fig.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart[ema],
        line=dict(color=color, width=1),
        name=name), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["RSI"],
    line=dict(color="#c678dd", width=1.5),
    name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash",
              line_color="#ff4466", row=2, col=1)
fig.add_hline(y=30, line_dash="dash",
              line_color="#00ff88", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["MACD"],
    line=dict(color="#00ff88", width=1.5),
    name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["MACD_signal"],
    line=dict(color="#ff4466", width=1.5),
    name="Signal"), row=3, col=1)
fig.add_bar(x=df_chart.index, y=df_chart["MACD_hist"],
            name="Histogramme",
            marker_color=["#00ff88" if v>=0 else "#ff4466"
                         for v in df_chart["MACD_hist"]],
            row=3, col=1)

fig.update_layout(
    height=700,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="white"),
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(bgcolor="#161b22", font=dict(color="white"))
)
fig.update_xaxes(gridcolor="#1e2d3d")
fig.update_yaxes(gridcolor="#1e2d3d")

st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown(f"""
<div style="text-align:center; color:#666; font-size:12px;">
🤖 XAUUSD ML System | Accuracy WF: 62.63% | HC: 66.95% |
Dernière mise à jour: {datetime.now(timezone.utc).strftime("%H:%M UTC")} |
⚠️ Pas un conseil financier
</div>
""", unsafe_allow_html=True)
"Fix features calculation"
