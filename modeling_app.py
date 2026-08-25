"""Four-page Streamlit app for sales exploration and predictive modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

st.set_page_config(page_title="顧客與訂單預測中心", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e5eaf1;
        border-radius:12px; padding:16px 18px; min-height:110px;}
    [data-testid="stMetricValue"] {font-size:clamp(1.55rem,2.2vw,2.5rem);}
    div[data-testid="stSidebar"] {background:#f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="正在載入與整理資料…")
def load_data() -> dict[str, pd.DataFrame]:
    customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(RAW / "order_items.csv")
    products = pd.read_csv(RAW / "products.csv").rename(columns={"unit_price": "list_price"})
    sessions = pd.read_csv(RAW / "sessions.csv", parse_dates=["session_start"])

    items["line_revenue"] = items["quantity"] * items["unit_price"] * (1-items["discount_rate"])
    facts = (items.merge(products, on="product_id", how="left", validate="many_to_one")
             .merge(orders, on="order_id", how="left", validate="many_to_one")
             .merge(customers, on="customer_id", how="left", validate="many_to_one"))
    facts = facts.loc[facts["status"].eq("completed")].copy()
    facts["month"] = facts["order_date"].dt.to_period("M").dt.to_timestamp()

    order_level = facts.groupby("order_id", as_index=False).agg(
        customer_id=("customer_id", "first"), order_date=("order_date", "first"),
        segment=("segment", "first"), acquisition_channel=("acquisition_channel", "first"),
        city=("city", "first"), payment_type=("payment_type", "first"),
        order_amount=("line_revenue", "sum"), units=("quantity", "sum"),
        avg_discount=("discount_rate", "mean"), product_count=("product_id", "nunique"))
    order_level["order_month"] = order_level["order_date"].dt.month
    order_level["order_quarter"] = order_level["order_date"].dt.quarter
    return {"customers": customers, "orders": orders, "sessions": sessions,
            "facts": facts, "order_level": order_level}


@st.cache_data(show_spinner="正在建立顧客特徵…")
def make_customer_features(_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers, orders, sessions = _data["customers"], _data["order_level"], _data["sessions"]
    cutoff = max(orders["order_date"].max(), sessions["session_start"].max())
    spend = orders.groupby("customer_id", as_index=False).agg(
        order_count=("order_id", "nunique"), total_spend=("order_amount", "sum"),
        avg_order_value=("order_amount", "mean"), total_units=("units", "sum"),
        last_order=("order_date", "max"))
    activity = sessions.groupby("customer_id", as_index=False).agg(
        session_count=("session_id", "nunique"), last_session=("session_start", "max"))
    frame = customers.merge(spend, on="customer_id", how="left").merge(activity, on="customer_id", how="left")
    numeric = ["order_count", "total_spend", "avg_order_value", "total_units", "session_count"]
    frame[numeric] = frame[numeric].fillna(0)
    frame["recency_days"] = (cutoff-frame["last_order"]).dt.days.fillna(999)
    frame["tenure_days"] = (cutoff-frame["signup_date"]).dt.days.clip(lower=0)
    frame["is_vip"] = frame["segment"].eq("vip").astype(int)
    return frame


CLASS_NUM = ["order_count", "total_spend", "avg_order_value", "total_units",
             "session_count", "recency_days", "tenure_days"]
CLASS_CAT = ["acquisition_channel", "city"]
REG_NUM = ["units", "avg_discount", "product_count", "order_month", "order_quarter"]
REG_CAT = ["segment", "acquisition_channel", "city", "payment_type"]


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


@st.cache_resource(show_spinner="正在訓練分類模型…")
def train_classifier(features: pd.DataFrame, model_name: str):
    x, y = features[CLASS_NUM+CLASS_CAT], features["is_vip"]
    estimator = (LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
                 if model_name == "LogisticRegression" else
                 RandomForestClassifier(n_estimators=350, max_depth=8, min_samples_leaf=4,
                                        class_weight="balanced", random_state=42, n_jobs=-1))
    model = Pipeline([("prep", preprocessor(CLASS_NUM, CLASS_CAT)), ("model", estimator)])
    xt, xv, yt, yv = train_test_split(x, y, test_size=.25, stratify=y, random_state=42)
    model.fit(xt, yt)
    probability = model.predict_proba(xv)[:, 1]
    predicted = probability >= .5
    scores = {
        "AUC": roc_auc_score(yv, probability),
        "準確率": accuracy_score(yv, predicted),
        "Precision": precision_score(yv, predicted, zero_division=0),
        "Recall": recall_score(yv, predicted, zero_division=0),
        "F1": f1_score(yv, predicted, zero_division=0),
    }
    model.fit(x, y)
    return model, scores


@st.cache_resource(show_spinner="正在訓練迴歸模型…")
def train_regressor(orders: pd.DataFrame, model_name: str):
    x, y = orders[REG_NUM+REG_CAT], orders["order_amount"]
    estimators = {
        "RandomForest": RandomForestRegressor(n_estimators=350, max_depth=12,
                                               min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=10.0, max_iter=10000),
    }
    model = Pipeline([("prep", preprocessor(REG_NUM, REG_CAT)), ("model", estimators[model_name])])
    xt, xv, yt, yv = train_test_split(x, y, test_size=.25, random_state=42)
    model.fit(xt, yt)
    prediction = np.maximum(model.predict(xv), 0)
    scores = {"MAE": mean_absolute_error(yv, prediction), "R²": r2_score(yv, prediction)}
    model.fit(x, y)
    return model, scores


def money(value: float) -> str:
    return f"NT$ {value:,.0f}"


data = load_data()
facts, order_level = data["facts"], data["order_level"]

with st.sidebar:
    st.title("🧭 顧客與訂單預測")
    page = st.radio("頁面", ["1｜營運總覽", "2｜資料檢視", "3｜分類預測", "4｜迴歸預測"])
    st.divider()
    st.caption(f"資料期間：{facts['order_date'].min():%Y-%m-%d} ～ {facts['order_date'].max():%Y-%m-%d}")


#---------------------------------------------------------------
if page == "1｜營運總覽":
    st.title("營運總覽")
    st.caption("從營收、訂單、顧客與流量快速掌握經營狀況")
    with st.sidebar:
        st.subheader("總覽篩選")
        segments = st.multiselect("客戶類型", sorted(facts["segment"].unique()),
                                  default=sorted(facts["segment"].unique()))
        channels = st.multiselect("獲客管道", sorted(facts["acquisition_channel"].unique()),
                                  default=sorted(facts["acquisition_channel"].unique()))
    shown = facts.loc[facts["segment"].isin(segments) & facts["acquisition_channel"].isin(channels)]
    if shown.empty:
        st.warning("目前篩選條件沒有資料。"); st.stop()
    shown_orders = shown.groupby("order_id", as_index=False)["line_revenue"].sum()
    k1, k2, k3 = st.columns(3, gap="large")
    k1.metric("銷售金額", money(shown["line_revenue"].sum()))
    k2.metric("完成訂單", f"{shown['order_id'].nunique():,}")
    k3.metric("平均客單價", money(shown_orders["line_revenue"].mean()))
    st.write("")

    left, right = st.columns([1.5, 1], gap="large")
    monthly = shown.groupby("month", as_index=False)["line_revenue"].sum()
    with left:
        st.subheader("月銷售金額趨勢")
        st.line_chart(monthly, x="month", y="line_revenue", color="#2563eb")
    with right:
        st.subheader("客戶類型營收")
        by_segment = shown.groupby("segment", as_index=False)["line_revenue"].sum()
        st.bar_chart(by_segment, x="segment", y="line_revenue", color="#7c3aed")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("獲客管道營收")
        channel_sales = shown.groupby("acquisition_channel", as_index=False)["line_revenue"].sum()
        st.bar_chart(channel_sales, x="acquisition_channel", y="line_revenue", color="#f59e0b")
    with right:
        st.subheader("產品類別營收")
        category_sales = shown.groupby("category", as_index=False)["line_revenue"].sum()
        st.bar_chart(category_sales, x="category", y="line_revenue", color="#0ea5e9")

#---------------------------------------------------------------
elif page == "2｜資料檢視":
    st.title("資料檢視")
    st.caption("使用頁籤切換三個核心分析主題")
    customer_tab, traffic_tab, monthly_tab = st.tabs([
        "客戶類型 × 訂單", "流量來源 × 訂單", "月份 × 訂單金額"])
    with customer_tab:
        customer_stats = order_level.groupby("segment", as_index=False).agg(
            訂單數=("order_id", "nunique"), 訂單金額=("order_amount", "sum"),
            平均訂單金額=("order_amount", "mean"), 顧客數=("customer_id", "nunique"))
        st.subheader("客戶類型、訂單數與訂單金額統計")
        left, right = st.columns(2)
        left.bar_chart(customer_stats, x="segment", y="訂單數", color="#14b8a6")
        right.bar_chart(customer_stats, x="segment", y="訂單金額", color="#8b5cf6")
        st.dataframe(customer_stats, hide_index=True, width="stretch",
                     column_config={"訂單金額":st.column_config.NumberColumn(format="NT$ %,.0f"),
                                    "平均訂單金額":st.column_config.NumberColumn(format="NT$ %,.0f")})
    with traffic_tab:
        sessions_by_customer = data["sessions"].groupby(["customer_id", "traffic_source"], as_index=False).agg(
            工作階段數=("session_id", "nunique"))
        orders_by_customer = order_level.groupby("customer_id", as_index=False).agg(訂單數=("order_id", "nunique"))
        traffic = sessions_by_customer.merge(orders_by_customer, on="customer_id", how="left").fillna({"訂單數":0})
        traffic_summary = traffic.groupby("traffic_source", as_index=False).agg(
            工作階段數=("工作階段數", "sum"), 訂單數=("訂單數", "sum"))
        correlation = traffic[["工作階段數", "訂單數"]].corr().iloc[0, 1]
        st.metric("工作階段數與訂單數相關係數", f"{correlation:.3f}")
        st.scatter_chart(traffic, x="工作階段數", y="訂單數", color="traffic_source", size=60)
        st.dataframe(traffic_summary, hide_index=True, width="stretch")
        st.caption("相關係數描述線性關聯，不代表流量來源造成訂單增加。")

    with monthly_tab:
        monthly_stats = order_level.assign(月份=order_level["order_date"].dt.to_period("M").dt.to_timestamp()).groupby(
            "月份", as_index=False).agg(訂單數=("order_id", "nunique"), 訂單金額=("order_amount", "sum"),
                                      平均訂單金額=("order_amount", "mean"))
        st.subheader("月份與訂單金額統計")
        st.line_chart(monthly_stats, x="月份", y="訂單金額", color="#2563eb")
        st.dataframe(monthly_stats, hide_index=True, width="stretch",
                     column_config={"月份":st.column_config.DateColumn(format="YYYY-MM"),
                                    "訂單金額":st.column_config.NumberColumn(format="NT$ %,.0f"),
                                    "平均訂單金額":st.column_config.NumberColumn(format="NT$ %,.0f")})

#---------------------------------------------------------------
elif page == "3｜分類預測":
    st.title("VIP 分類預測")
    st.caption("選擇模型並輸入顧客資料，預測成為 VIP 的可能性")
    features = make_customer_features(data)
    st.subheader("模型選擇")
    model_name = st.radio("選擇分類演算法", ["LogisticRegression", "RandomForest"],
                          horizontal=True, label_visibility="collapsed")
    trained_models = {}
    comparison_rows = []
    for candidate in ["LogisticRegression", "RandomForest"]:
        candidate_model, candidate_scores = train_classifier(features, candidate)
        trained_models[candidate] = candidate_model
        comparison_rows.append({"模型":candidate, **candidate_scores})
    model = trained_models[model_name]
    comparison = pd.DataFrame(comparison_rows)
    st.subheader("模型能力測試（固定 25% 測試集）")
    st.dataframe(
        comparison.style.highlight_max(subset=["AUC", "準確率", "Precision", "Recall", "F1"],
                                       color="#dcfce7"),
        hide_index=True, width="stretch",
        column_config={name:st.column_config.NumberColumn(format="%.3f")
                       for name in ["AUC", "準確率", "Precision", "Recall", "F1"]},
    )
    selected_scores = comparison.loc[comparison["模型"].eq(model_name)].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{model_name} · AUC", f"{selected_scores['AUC']:.3f}")
    m2.metric("F1", f"{selected_scores['F1']:.3f}")
    m3.metric("Recall", f"{selected_scores['Recall']:.3f}")
    st.divider()
    st.subheader("輸入顧客資料")
    defaults = features.median(numeric_only=True)
    a, b, c = st.columns(3)
    order_count = a.number_input("歷史訂單數", 0, 1000, int(defaults["order_count"]), key="vip_order_count")
    total_spend = b.number_input("累計消費金額", 0.0, value=float(defaults["total_spend"]),
                                 step=1000.0, key="vip_total_spend")
    avg_order_value = total_spend/order_count if order_count > 0 else 0.0
    c.metric("平均訂單金額（自動計算）", money(avg_order_value),
             help="累計消費金額 ÷ 歷史訂單數；訂單數為 0 時顯示 0。")
    values = {
        "order_count":order_count,
        "total_spend":total_spend,
        "avg_order_value":avg_order_value,
        "total_units":a.number_input("累計購買件數", 0, 10000, int(defaults["total_units"])),
        "session_count":b.number_input("網站工作階段數", 0, 10000, int(defaults["session_count"])),
        "recency_days":c.number_input("距最近購買天數", 0, 3650, int(defaults["recency_days"])),
        "tenure_days":a.number_input("加入天數", 0, 10000, int(defaults["tenure_days"])),
        "acquisition_channel":b.selectbox("獲客管道", sorted(features["acquisition_channel"].unique())),
        "city":c.selectbox("城市", sorted(features["city"].unique())),
    }
    submit = st.button("預測 VIP 機率", type="primary", width="stretch")
    if submit:
        probability = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
        st.progress(probability, text=f"{model_name} 預測 VIP 機率：{probability:.1%}")
        if probability >= .7:
            st.success("高潛力 VIP")
        else:
            st.info("可持續培養顧客價值")
    st.caption("VIP 標籤定義：customers.csv 中 segment = vip；預測結果供決策參考，不代表因果關係。")

#---------------------------------------------------------------
elif page == "4｜迴歸預測":
    st.title("訂單金額迴歸預測")
    st.caption("選擇模型並輸入訂單資訊，預測該筆訂單可能的金額")
    with st.sidebar:
        model_name = st.selectbox("迴歸模型", ["RandomForest", "Ridge", "Lasso"])
    model, scores = train_regressor(order_level, model_name)
    m1, m2 = st.columns(2)
    m1.metric("測試集 MAE", money(scores["MAE"]))
    m2.metric("測試集 R²", f"{scores['R²']:.3f}")
    with st.form("regression_form"):
        a, b, c = st.columns(3)
        reg_values = {
            "units":a.number_input("購買件數", 1, 1000, int(order_level["units"].median())),
            "avg_discount":b.slider("平均折扣率", 0.0, 1.0, float(order_level["avg_discount"].median()), .01),
            "product_count":c.number_input("產品種類數", 1, 100, int(order_level["product_count"].median())),
            "order_month":a.selectbox("訂單月份", list(range(1,13))),
            "order_quarter":b.selectbox("訂單季度", [1,2,3,4]),
            "segment":c.selectbox("客戶類型", sorted(order_level["segment"].unique())),
            "acquisition_channel":a.selectbox("獲客管道", sorted(order_level["acquisition_channel"].unique())),
            "city":b.selectbox("城市", sorted(order_level["city"].unique())),
            "payment_type":c.selectbox("付款方式", sorted(order_level["payment_type"].unique())),
        }
        submit = st.form_submit_button("預測訂單金額", type="primary", width="stretch")
    if submit:
        prediction = max(float(model.predict(pd.DataFrame([reg_values]))[0]), 0)
        st.subheader("預測結果")
        st.metric(f"{model_name} 預測訂單金額", money(prediction))
        st.info(f"模型測試集平均絕對誤差約為 {money(scores['MAE'])}，請將預測視為估計值。")


