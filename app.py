
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XAUUSD ML Scanner",
    page_icon="🥇",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; }
h1, h2, h3, p, div { color: #ffffff; }
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

if model_xgb is None:
    st.stop()

# ============================================
# TÉLÉCHARGEMENT DONNÉES
# ============================================
@st.cache_data(ttl=900)
def download_ticker(ticker, interval="1h", period="60d"):
    try:
        df = yf.download(ticker, interval=interval,
                        period=period, progress=False)
        # Correction multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        # S'assurer que les colonnes existent
        needed = ["Open","High","Low","Close","Volume"]
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan
        return df[needed].copy()
    except Exception as e:
        return None

@st.cache_data(ttl=900)
def get_all_data():
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
        df = download_ticker(ticker)
        if df is not None and len(df) > 10:
            data[nom] = df
    return data

# ============================================
# PRÉPARATION FEATURES
# ============================================
@st.cache_data(ttl=900)
def prepare_features(_data):
    try:
        if "GOLD" not in _data:
            return None

        df = _data["GOLD"].copy()

        # Ajouter actifs macro
        for nom in ["DXY","VIX","US10Y","SILVER","SP500"]:
            if nom in _data:
                tmp = _data[nom][["Close"]].rename(
                    columns={"Close": nom})
                df = pd.merge_asof(
                    df.sort_index(),
                    tmp.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward")

        df = df.ffill().dropna()

        if len(df) < 200:
            return None

        # Resample H4 et D1
        def resample_tf(df_in, tf):
            return df_in[["Open","High","Low",
                          "Close","Volume"]].resample(tf).agg({
                "Open":"first","High":"max",
                "Low":"min","Close":"last",
                "Volume":"sum"}).dropna()

        df_h4 = resample_tf(df, "4h")
        df_d1 = resample_tf(df, "1D")

        # EMA H4
        df_h4["EMA20_H4"] = ta.trend.ema_indicator(df_h4["Close"],20)
        df_h4["EMA50_H4"] = ta.trend.ema_indicator(df_h4["Close"],50)
        df_h4["RSI_H4"]   = ta.momentum.rsi(df_h4["Close"],14)
        df_h4["Trend_H4"] = (df_h4["EMA20_H4"]>df_h4["EMA50_H4"]).astype(int)

        # EMA D1
        df_d1["EMA20_D1"]  = ta.trend.ema_indicator(df_d1["Close"],20)
        df_d1["EMA50_D1"]  = ta.trend.ema_indicator(df_d1["Close"],50)
        df_d1["EMA200_D1"] = ta.trend.ema_indicator(df_d1["Close"],200)
        df_d1["RSI_D1"]    = ta.momentum.rsi(df_d1["Close"],14)
        df_d1["Trend_D1"]  = (df_d1["EMA20_D1"]>df_d1["EMA50_D1"]).astype(int)

        # Merge H4 et D1
        df = pd.merge_asof(df.sort_index(),
            df_h4[["Trend_H4","EMA20_H4","EMA50_H4","RSI_H4"]].sort_index(),
            left_index=True, right_index=True, direction="backward")
        df = pd.merge_asof(df.sort_index(),
            df_d1[["Trend_D1","EMA20_D1","EMA50_D1",
                   "EMA200_D1","RSI_D1"]].sort_index(),
            left_index=True, right_index=True, direction="backward")

        # RSI H1
        df["RSI"]    = ta.momentum.rsi(df["Close"],14)
        df["RSI_ob"] = (df["RSI"]>70).astype(int)
        df["RSI_os"] = (df["RSI"]<30).astype(int)

        # EMA H1
        df["EMA20"]  = ta.trend.ema_indicator(df["Close"],20)
        df["EMA50"]  = ta.trend.ema_indicator(df["Close"],50)
        df["EMA200"] = ta.trend.ema_indicator(df["Close"],200)

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"]   = macd.macd_diff()
        df["MACD_cross_up"]   = ((df["MACD"]>df["MACD_signal"])&
            (df["MACD"].shift(1)<df["MACD_signal"].shift(1))).astype(int)
        df["MACD_cross_down"] = ((df["MACD"]<df["MACD_signal"])&
            (df["MACD"].shift(1)>df["MACD_signal"].shift(1))).astype(int)

        # Bollinger
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = df["BB_upper"]-df["BB_lower"]
        df["BB_pct"]   = (df["Close"]-df["BB_lower"])/df["BB_width"]

        # ATR
        df["ATR"] = ta.volatility.average_true_range(
            df["High"],df["Low"],df["Close"],14)

        # Returns
        df["Return_1h"]  = df["Close"].pct_change(1)
        df["Return_4h"]  = df["Close"].pct_change(4)
        df["Return_24h"] = df["Close"].pct_change(24)

        # Distance EMA
        df["Price_vs_EMA20"]  = (df["Close"]-df["EMA20"])/df["EMA20"]*100
        df["Price_vs_EMA50"]  = (df["Close"]-df["EMA50"])/df["EMA50"]*100
        df["Price_vs_EMA200"] = (df["Close"]-df["EMA200"])/df["EMA200"]*100
        df["EMA20_vs_EMA50"]  = (df["EMA20"]-df["EMA50"])/df["EMA50"]*100
        df["High_Low_pct"]    = (df["High"]-df["Low"])/df["Close"]*100

        # Sessions
        df["Hour"]            = df.index.hour
        df["Session_London"]  = ((df["Hour"]>=7)&(df["Hour"]<16)).astype(int)
        df["Session_NY"]      = ((df["Hour"]>=13)&(df["Hour"]<21)).astype(int)
        df["Session_Overlap"] = ((df["Hour"]>=13)&(df["Hour"]<16)).astype(int)
        df["Session_Asian"]   = ((df["Hour"]>=0)&(df["Hour"]<7)).astype(int)

        # Macro returns
        if "DXY" in df.columns:
            df["DXY_return"]  = df["DXY"].pct_change(4)
        else:
            df["DXY_return"]  = 0
        if "VIX" in df.columns:
            df["VIX_return"]  = df["VIX"].pct_change(4)
        else:
            df["VIX_return"]  = 0
        if "SILVER" in df.columns:
            df["SILVER_return"] = df["SILVER"].pct_change(4)
        else:
            df["SILVER_return"] = 0
        if "SP500" in df.columns:
            df["SP500_return"]  = df["SP500"].pct_change(4)
        else:
            df["SP500_return"]  = 0

        df["Gold_DXY_div"]   = df["Return_4h"] + df["DXY_return"]
        df["Trend_bull_3TF"] = ((df["Trend_D1"]==1)&
                                 (df["Trend_H4"]==1)).astype(int)
        df["Trend_bear_3TF"] = ((df["Trend_D1"]==0)&
                                 (df["Trend_H4"]==0)).astype(int)

        # SMC
        df["BOS_bull"] = (df["High"]>df["High"].shift(1).rolling(10).max()).astype(int)
        df["BOS_bear"] = (df["Low"]<df["Low"].shift(1).rolling(10).min()).astype(int)
        df["Higher_High"] = (df["High"]>df["High"].shift(1)).astype(int)
        df["Lower_Low"]   = (df["Low"]<df["Low"].shift(1)).astype(int)
        df["CHoCH_bear"]  = ((df["Higher_High"].shift(3)==1)&
                              (df["Higher_High"].shift(2)==1)&
                              (df["Lower_Low"]==1)).astype(int)
        df["CHoCH_bull"]  = ((df["Lower_Low"].shift(3)==1)&
                              (df["Lower_Low"].shift(2)==1)&
                              (df["Higher_High"]==1)).astype(int)
        df["Bearish_candle"] = (df["Close"]<df["Open"]).astype(int)
        df["Bullish_candle"] = (df["Close"]>df["Open"]).astype(int)
        df["Strong_up"]   = (df["Return_1h"]>df["ATR"]/df["Close"]*100).astype(int)
        df["Strong_down"] = (df["Return_1h"]<-df["ATR"]/df["Close"]*100).astype(int)
        df["OB_bull"] = ((df["Bearish_candle"].shift(1)==1)&
                          (df["Strong_up"]==1)).astype(int)
        df["OB_bear"] = ((df["Bullish_candle"].shift(1)==1)&
                          (df["Strong_down"]==1)).astype(int)
        df["Range_high"]  = df["High"].rolling(100).max()
        df["Range_low"]   = df["Low"].rolling(100).min()
        df["Range_mid"]   = (df["Range_high"]+df["Range_low"])/2
        df["In_discount"] = (df["Close"]<df["Range_mid"]).astype(int)
        df["In_premium"]  = (df["Close"]>df["Range_mid"]).astype(int)
        tol = 0.001
        df["Equal_highs"] = (abs(df["High"]-df["High"].shift(1))/df["High"]<tol).astype(int)
        df["Equal_lows"]  = (abs(df["Low"]-df["Low"].shift(1))/df["Low"]<tol).astype(int)
        df["SMC_bull_score"]   = (df["BOS_bull"]+df["CHoCH_bull"]+
                                   df["OB_bull"]+df["In_discount"])
        df["SMC_bear_score"]   = (df["BOS_bear"]+df["CHoCH_bear"]+
                                   df["OB_bear"]+df["In_premium"])
        df["SMC_bull_confirm"] = ((df["SMC_bull_score"]>=2)&
                                   (df["Trend_bull_3TF"]==1)).astype(int)
        df["SMC_bear_confirm"] = ((df["SMC_bear_score"]>=2)&
                                   (df["Trend_bear_3TF"]==1)).astype(int)

        # Indicateurs avancés
        stoch = ta.momentum.StochasticOscillator(
            df["High"],df["Low"],df["Close"],14)
        df["Stoch_K"]          = stoch.stoch()
        df["Stoch_D"]          = stoch.stoch_signal()
        df["Stoch_ob"]         = (df["Stoch_K"]>80).astype(int)
        df["Stoch_os"]         = (df["Stoch_K"]<20).astype(int)
        df["Stoch_cross_up"]   = ((df["Stoch_K"]>df["Stoch_D"])&
            (df["Stoch_K"].shift(1)<df["Stoch_D"].shift(1))).astype(int)
        df["Stoch_cross_down"] = ((df["Stoch_K"]<df["Stoch_D"])&
            (df["Stoch_K"].shift(1)>df["Stoch_D"].shift(1))).astype(int)
        df["Williams_R"]  = ta.momentum.williams_r(
            df["High"],df["Low"],df["Close"],14)
        df["Williams_ob"] = (df["Williams_R"]>-20).astype(int)
        df["Williams_os"] = (df["Williams_R"]<-80).astype(int)
        df["CCI"]    = ta.trend.cci(df["High"],df["Low"],df["Close"],20)
        df["CCI_ob"] = (df["CCI"]>100).astype(int)
        df["CCI_os"] = (df["CCI"]<-100).astype(int)

        # Ichimoku
        ichi = ta.trend.IchimokuIndicator(df["High"],df["Low"],9,26,52)
        df["Ichi_tenkan"]   = ichi.ichimoku_conversion_line()
        df["Ichi_kijun"]    = ichi.ichimoku_base_line()
        df["Ichi_spanA"]    = ichi.ichimoku_a()
        df["Ichi_spanB"]    = ichi.ichimoku_b()
        df["Ichi_bull"]     = (df["Close"]>df["Ichi_spanA"]).astype(int)
        df["Ichi_tk_cross"] = (df["Ichi_tenkan"]>df["Ichi_kijun"]).astype(int)

        # VWAP
        df["VWAP"]          = (df["Close"]*df["Volume"]).cumsum()/df["Volume"].cumsum()
        df["Price_vs_VWAP"] = (df["Close"]-df["VWAP"])/df["VWAP"]*100
        df["Above_VWAP"]    = (df["Close"]>df["VWAP"]).astype(int)

        # OBV
        df["OBV"]            = ta.volume.on_balance_volume(df["Close"],df["Volume"])
        df["OBV_EMA"]        = ta.trend.ema_indicator(df["OBV"],20)
        df["OBV_trend"]      = (df["OBV"]>df["OBV_EMA"]).astype(int)
        df["OBV_divergence"] = ((df["Close"]>df["Close"].shift(5))&
                                  (df["OBV"]<df["OBV"].shift(5))).astype(int)

        # Pivot Points
        df_d1_p = resample_tf(df,"1D")
        df_d1_p["PP"] = (df_d1_p["High"]+df_d1_p["Low"]+df_d1_p["Close"])/3
        df_d1_p["R1"] = 2*df_d1_p["PP"]-df_d1_p["Low"]
        df_d1_p["S1"] = 2*df_d1_p["PP"]-df_d1_p["High"]
        df_d1_p["R2"] = df_d1_p["PP"]+(df_d1_p["High"]-df_d1_p["Low"])
        df_d1_p["S2"] = df_d1_p["PP"]-(df_d1_p["High"]-df_d1_p["Low"])
        df_d1_p["R3"] = df_d1_p["High"]+2*(df_d1_p["PP"]-df_d1_p["Low"])
        df_d1_p["S3"] = df_d1_p["Low"]-2*(df_d1_p["High"]-df_d1_p["PP"])
        pivot = df_d1_p[["PP","R1","S1","R2","S2","R3","S3"]].shift(1)
        df = pd.merge_asof(df.sort_index(),pivot.sort_index(),
                           left_index=True,right_index=True,
                           direction="backward")
        df["Dist_PP"]  = (df["Close"]-df["PP"])/df["Close"]*100
        df["Dist_R1"]  = (df["R1"]-df["Close"])/df["Close"]*100
        df["Dist_S1"]  = (df["Close"]-df["S1"])/df["Close"]*100
        df["Near_PP"]  = (abs(df["Dist_PP"])<0.1).astype(int)
        df["Near_R1"]  = (abs(df["Dist_R1"])<0.1).astype(int)
        df["Near_S1"]  = (abs(df["Dist_S1"])<0.1).astype(int)
        df["Above_PP"] = (df["Close"]>df["PP"]).astype(int)

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
            (df["Close"]>df["Open"])&(df["Close"]>df["Open"].shift(1))&
            (df["Open"]<df["Close"].shift(1))).astype(int)
        df["Bear_engulf"]   = ((df["Close"].shift(1)>df["Open"].shift(1))&
            (df["Close"]<df["Open"])&(df["Close"]<df["Open"].shift(1))&
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
        df["Wyckoff_accum"]      = ((df["Close"]<df["Wyckoff_range_low"]+
            df["Wyckoff_range"]*0.3)&
            (df["Volume"]>df["Volume"].rolling(20).mean())).astype(int)
        df["Wyckoff_distrib"]    = ((df["Close"]>df["Wyckoff_range_high"]-
            df["Wyckoff_range"]*0.3)&
            (df["Volume"]>df["Volume"].rolling(20).mean())).astype(int)

        # Momentum institutionnel
        df["Inst_momentum"]      = (df["Close"].rolling(20).mean()-
                                     df["Close"].rolling(50).mean())
        df["Inst_bull"]          = (df["Inst_momentum"]>0).astype(int)
        df["Inst_bear"]          = (df["Inst_momentum"]<0).astype(int)
        df["Vol_price_div_bull"] = ((df["Close"]<df["Close"].shift(5))&
            (df["Volume"]>df["Volume"].rolling(20).mean()*1.5)).astype(int)
        df["Vol_price_div_bear"] = ((df["Close"]>df["Close"].shift(5))&
            (df["Volume"]>df["Volume"].rolling(20).mean()*1.5)).astype(int)

        # MSS + Inducement
        df["MSS_bull"]        = ((df["BOS_bull"]==1)&(df["CHoCH_bull"]==1)).astype(int)
        df["MSS_bear"]        = ((df["BOS_bear"]==1)&(df["CHoCH_bear"]==1)).astype(int)
        df["Inducement_bull"] = ((df["Equal_lows"]==1)&(df["In_discount"]==1)).astype(int)
        df["Inducement_bear"] = ((df["Equal_highs"]==1)&(df["In_premium"]==1)).astype(int)

        # Score global
        df["Global_bull_score"] = (df["SMC_bull_score"]+df["Inst_bull"]+
            df["OBV_trend"]+df["Above_VWAP"]+df["Above_PP"]+
            df["Ichi_bull"]+df["Wyckoff_accum"]+df["Bull_engulf"]+
            df["Hammer"]+df["Trend_bull_3TF"])
        df["Global_bear_score"] = (df["SMC_bear_score"]+df["Inst_bear"]+
            (1-df["OBV_trend"])+(1-df["Above_VWAP"])+(1-df["Above_PP"])+
            (1-df["Ichi_bull"])+df["Wyckoff_distrib"]+df["Bear_engulf"]+
            df["Shooting_star"]+df["Trend_bear_3TF"])
        df["Global_net_score"]  = df["Global_bull_score"]-df["Global_bear_score"]

        # Sentiment + Calendrier (valeurs par défaut)
        df["Sentiment_NLP"]   = 0.0
        df["Sentiment_bull"]  = 0
        df["Sentiment_bear"]  = 0
        df["News_dans_30min"] = 0
        df["News_dans_60min"] = 0
        df["Surprise_bull"]   = 0
        df["Surprise_bear"]   = 0
        df["Macro_score_cal"] = 0

        # FRED features (valeurs par défaut)
        fred_features = ["NFP_change","CPI_change","GDP_change",
                        "FED_RATE_change","PPI_change","UNEMP_change",
                        "REAL_RATE_change","BREAKEVEN_change",
                        "Macro_bull_gold","Macro_bear_gold","Macro_net_score"]
        for f in fred_features:
            df[f] = 0.0

        df = df.dropna()

        if len(df) == 0:
            return None

        return df

    except Exception as e:
        st.error(f"Erreur features: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================
# SCORING
# ============================================
def get_score(df_app, model_xgb, model_lgb, model_xgb2, features):
    try:
        # Features disponibles
        feat_ok = [f for f in features if f in df_app.columns]
        feat_missing = [f for f in features if f not in df_app.columns]

        if feat_missing:
            # Ajouter features manquantes avec valeur 0
            for f in feat_missing:
                df_app[f] = 0.0

        X = df_app[features].iloc[[-1]]
        X = X.fillna(0)

        p1 = model_xgb.predict_proba(X)[0][1]
        p2 = model_lgb.predict_proba(X)[0][1]
        p3 = model_xgb2.predict_proba(X)[0][1]
        return (p1+p2+p3)/3, p1, p2, p3

    except Exception as e:
        st.error(f"Erreur scoring: {e}")
        return 0.5, 0.5, 0.5, 0.5

def get_sentiment():
    try:
        analyzer = SentimentIntensityAnalyzer()
        urls = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
            "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
        ]
        titres = []
        for url in urls:
            try:
                resp = requests.get(url,timeout=5,
                    headers={"User-Agent":"Mozilla/5.0"})
                soup = BeautifulSoup(resp.content,"lxml-xml")
                for item in soup.find_all("title")[1:8]:
                    if item.text.strip():
                        titres.append(item.text.strip())
            except:
                pass
        if not titres:
            return 0.0, []
        scores = [analyzer.polarity_scores(t)["compound"] for t in titres]
        return sum(scores)/len(scores), titres
    except:
        return 0.0, []

def get_calendar():
    events = []
    news_30 = False
    news_60 = False
    try:
        for url in ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"]:
            try:
                resp = requests.get(url,timeout=8,
                    headers={"User-Agent":"Mozilla/5.0"})
                if resp.status_code==200 and len(resp.content)>100:
                    data = resp.json()
                    now  = datetime.now(timezone.utc)
                    for ev in data:
                        if ev.get("impact") not in ["High","Medium"]:
                            continue
                        try:
                            t    = datetime.fromisoformat(
                                ev["date"].replace("Z","+00:00"))
                            diff = (t-now).total_seconds()/60
                            events.append({
                                "title" :ev.get("title",""),
                                "impact":ev.get("impact",""),
                                "diff"  :diff})
                            if 0<diff<=30: news_30=True
                            if 0<diff<=60: news_60=True
                        except:
                            continue
                    break
            except:
                continue
    except:
        pass
    return events, news_30, news_60

def calculer_position(proba, capital, atr, prix, kelly_frac=0.126, rr=2.0):
    if proba>=0.85:   risk_pct=min(kelly_frac*2.0,0.03)
    elif proba>=0.80: risk_pct=min(kelly_frac*1.5,0.025)
    elif proba>=0.75: risk_pct=min(kelly_frac*1.0,0.02)
    elif proba>=0.70: risk_pct=min(kelly_frac*0.75,0.015)
    else:             risk_pct=min(kelly_frac*0.5,0.01)
    sl = 1.5*atr
    return {
        "risk_pct"   :risk_pct*100,
        "risk_amount":capital*risk_pct,
        "sl_long"    :prix-sl,
        "tp_long"    :prix+sl*rr,
        "sl_short"   :prix+sl,
        "tp_short"   :prix-sl*rr,
        "sl_dist"    :sl
    }

# ============================================
# INTERFACE
# ============================================
st.title("🥇 XAUUSD ML Trading Scanner")
st.caption("Système ML — SMC + Technique + Macro + Sentiment")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    capital    = st.number_input("Capital (€)",100,100000,500,100)
    seuil_conf = st.slider("Seuil confiance",0.60,0.90,0.65,0.05)
    if st.button("🔄 Rafraîchir",type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**📊 Performances :**")
    st.metric("Walk-Forward","62.63%")
    st.metric("Haute Confiance","66.95%")
    st.metric("Fenêtres 7/7","> 55%")

# Chargement
with st.spinner("📥 Chargement données..."):
    market_data = get_all_data()

if not market_data or "GOLD" not in market_data:
    st.error("❌ Impossible de charger Gold")
    st.stop()

with st.spinner("🔧 Calcul features..."):
    df_app = prepare_features(market_data)

if df_app is None or len(df_app)==0:
    st.error("❌ Erreur dans le calcul des features — données insuffisantes")
    st.stop()

# Données actuelles
derniere    = df_app.iloc[-1]
prix        = float(derniere["Close"])
atr         = float(derniere["ATR"])
timestamp   = df_app.index[-1]

# Score ML
proba, p1, p2, p3 = get_score(
    df_app, model_xgb, model_lgb, model_xgb2, features)
proba_long  = proba
proba_short = 1-proba

# Signal
if proba_long>seuil_conf:
    signal="LONG"; color="#00ff88"; emoji="📈"
elif proba_short>seuil_conf:
    signal="SHORT"; color="#ff4466"; emoji="📉"
else:
    signal="ATTENDRE"; color="#ffaa00"; emoji="⏸️"

# Sentiment + Calendrier
sent_score, news_titres = get_sentiment()
events, news_30, news_60 = get_calendar()
pos = calculer_position(
    max(proba_long,proba_short),
    capital, atr, prix,
    config.get("kelly_frac",0.126))

# ============================================
# AFFICHAGE
# ============================================
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.metric("💰 Prix XAUUSD", f"{prix:.2f}",
              f"{derniere.get('Return_1h',0)*100:.2f}%")
    st.metric("⏰ Heure UTC",
              timestamp.strftime("%H:%M"))
    st.metric("📅 Date",
              timestamp.strftime("%Y-%m-%d"))

with col2:
    score_val = max(proba_long,proba_short)*100
    st.markdown(f"""
    <div style="background:#161b22;border:3px solid {color};
    border-radius:15px;padding:30px;text-align:center;">
    <h1 style="color:{color};font-size:52px;margin:0;">
    {emoji} {signal}</h1>
    <h2 style="color:{color};margin:5px;">
    Score : {score_val:.1f}%</h2>
    <p style="color:#888;margin:0;">
    XGB:{p1*100:.1f}% | LGB:{p2*100:.1f}% | XGB2:{p3*100:.1f}%
    </p></div>""", unsafe_allow_html=True)

with col3:
    st.metric("📈 Long",  f"{proba_long*100:.1f}%")
    st.metric("📉 Short", f"{proba_short*100:.1f}%")
    if news_30:
        st.error("🚨 NEWS 30min!")
    elif news_60:
        st.warning("⚠️ NEWS 60min")
    else:
        st.success("✅ Pas de news")

st.divider()

# SL/TP + Indicateurs
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🎯 SL/TP (ATR)")
    st.markdown(f"""
**Capital :** {capital}€
**Risque :** {pos["risk_pct"]:.2f}% = **{pos["risk_amount"]:.2f}€**
**ATR :** {atr:.2f}

📈 **LONG** — SL:`{pos["sl_long"]:.2f}` TP:`{pos["tp_long"]:.2f}`

📉 **SHORT** — SL:`{pos["sl_short"]:.2f}` TP:`{pos["tp_short"]:.2f}`
    """)

with col5:
    st.subheader("📊 Indicateurs")
    rsi = float(derniere.get("RSI",50))
    rsi_h4 = float(derniere.get("RSI_H4",50))
    rsi_d1 = float(derniere.get("RSI_D1",50))
    st.metric("RSI H1", f"{rsi:.1f}",
              "Surachat" if rsi>70 else ("Survente" if rsi<30 else "Neutre"))
    st.metric("RSI H4", f"{rsi_h4:.1f}")
    st.metric("RSI D1", f"{rsi_d1:.1f}")
    trend_h4 = int(derniere.get("Trend_H4",0))
    trend_d1 = int(derniere.get("Trend_D1",0))
    st.metric("H4", "🟢 Haussier" if trend_h4 else "🔴 Baissier")
    st.metric("D1", "🟢 Haussier" if trend_d1 else "🔴 Baissier")

with col6:
    st.subheader("📰 Sentiment")
    sent_label = ("🟢 POSITIF" if sent_score>0.05
                  else "🔴 NÉGATIF" if sent_score<-0.05
                  else "⚪ NEUTRE")
    st.metric("Sentiment Gold", f"{sent_score:.3f}", sent_label)
    analyzer = SentimentIntensityAnalyzer()
    for titre in news_titres[:4]:
        s = analyzer.polarity_scores(titre)["compound"]
        e = "🟢" if s>0.05 else ("🔴" if s<-0.05 else "⚪")
        st.markdown(f"{e} {titre[:50]}")

st.divider()

# Calendrier
st.subheader("📅 Calendrier Économique")
if events:
    cols_cal = st.columns(min(len(events[:6]),3))
    for i, ev in enumerate(events[:6]):
        diff = ev["diff"]
        emoji_ev = "🔴" if ev["impact"]=="High" else "🟡"
        if diff>0:
            h,m = int(diff//60),int(diff%60)
            timing = f"Dans {h}h{m:02d}m"
        else:
            timing = f"Il y a {abs(diff):.0f}min"
        with cols_cal[i%3]:
            st.markdown(f"{emoji_ev} **{ev['title'][:25]}**")
            st.caption(timing)
else:
    st.info("Pas d'événements majeurs")

st.divider()

# Graphique
st.subheader("📈 XAUUSD H1 — 100 dernières bougies")
df_chart = df_app.tail(100)

fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    row_heights=[0.7,0.3],
                    subplot_titles=["Prix + EMA","RSI"])

fig.add_trace(go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"],high=df_chart["High"],
    low=df_chart["Low"],close=df_chart["Close"],
    increasing_line_color="#00ff88",
    decreasing_line_color="#ff4466",
    name="XAUUSD"),row=1,col=1)

for ema,color_e,name_e in [
    ("EMA20","#58a6ff","EMA20"),
    ("EMA50","#ff8800","EMA50"),
    ("EMA200","#ff4466","EMA200")]:
    if ema in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index,y=df_chart[ema],
            line=dict(color=color_e,width=1),
            name=name_e),row=1,col=1)

fig.add_trace(go.Scatter(
    x=df_chart.index,y=df_chart["RSI"],
    line=dict(color="#c678dd",width=1.5),
    name="RSI"),row=2,col=1)
fig.add_hline(y=70,line_dash="dash",
              line_color="#ff4466",row=2,col=1)
fig.add_hline(y=30,line_dash="dash",
              line_color="#00ff88",row=2,col=1)

fig.update_layout(
    height=600,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="white"),
    xaxis_rangeslider_visible=False,
    showlegend=True)
fig.update_xaxes(gridcolor="#1e2d3d")
fig.update_yaxes(gridcolor="#1e2d3d")

st.plotly_chart(fig,use_container_width=True)

st.caption(f"⚠️ Pas un conseil financier | "
           f"Mise à jour: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
