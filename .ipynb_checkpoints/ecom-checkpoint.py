# streamlit_recommender_app.py
# Interactive Streamlit app for product recommendations, targeted ads and promotions
# Assumes the following files exist in the working directory (or change the paths):
# - ecom_synthetic/customers.csv
# - ecom_synthetic/ad_click_model_rf.joblib
# - ecom_synthetic/recommender_svd.joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

st.set_page_config(page_title="E‑commerce Recommender Demo", layout="wide")

st.title("E‑commerce & Digital Marketing")
st.title("Product Recommendations, Targeted Ads & Promotions")

# --- Paths ---
BASE = Path("ecom_synthetic")
CUSTOMER_CSV = BASE / "customers.csv"
RFC_PATH = BASE / "ad_click_model_rf.joblib"
SVD_PATH = BASE / "recommender_svd.joblib"

# --- Load data and models (with helpful user feedback) ---
@st.cache_resource
def load_models_and_data():
    data = None
    rfc = None
    svd = None
    if CUSTOMER_CSV.exists():
        data = pd.read_csv(CUSTOMER_CSV, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(CUSTOMER_CSV, nrows=1).columns else [])
    else:
        st.warning(f"{CUSTOMER_CSV} not found. Upload or place file in this folder.")

    if RFC_PATH.exists():
        rfc = joblib.load(RFC_PATH)
    else:
        st.warning(f"{RFC_PATH} not found. The ad-click model won't be available.")

    if SVD_PATH.exists():
        svd = joblib.load(SVD_PATH)
    else:
        st.warning(f"{SVD_PATH} not found. The recommender won't be available.")

    return data, rfc, svd

with st.spinner("Loading data and models..."):
    data, rfc, svd = load_models_and_data()

if data is None:
    st.stop()

# --- Basic data display ---
st.sidebar.header("Data & Models")
st.sidebar.write(f"Rows: {len(data)}")
st.sidebar.write(f"Unique users: {data['user_id'].nunique()}")
st.sidebar.write(f"Unique products: {data['product_id'].nunique()}")

# --- Prepare user_item matrix for SVD-based recommender ---
@st.cache_data
def build_user_item(df):
    df2 = df.copy()
    if 'clicked' not in df2.columns:
        df2['clicked'] = 0
    if 'purchased' not in df2.columns:
        df2['purchased'] = 0
    df2['weight'] = 0.05 + df2['clicked'] * 0.5 + df2['purchased'] * 2.5
    ui = df2.groupby(['user_id', 'product_id'])['weight'].sum().unstack(fill_value=0)
    return ui

user_item = build_user_item(data)

if svd is not None:
    svd_n_items = svd.components_.shape[1]
    if user_item.shape[1] > svd_n_items:
        keep = list(user_item.columns)[:svd_n_items]
        user_item = user_item[keep]
    elif user_item.shape[1] < svd_n_items:
        for i in range(svd_n_items - user_item.shape[1]):
            user_item[f"__pad_{i}"] = 0

user_factors = None
item_factors = None
if svd is not None:
    try:
        user_factors = svd.transform(user_item)
        item_factors = svd.components_.T
    except Exception as e:
        st.error("Error applying saved SVD to current data: " + str(e))

def recommend_for_user(user_id, n=6):
    if user_factors is None:
        return []
    if user_id not in list(user_item.index):
        return []
    ui = list(user_item.index).index(user_id)
    uvec = user_factors[ui]
    scores = item_factors.dot(uvec)
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    ranked = sorted(zip(list(user_item.columns), scores), key=lambda x: x[1], reverse=True)
    recs = [pid for pid, sc in ranked if pid not in seen][:n]
    return recs

def score_candidates_with_rfc(user_id, candidates, rfc_model, df):
    if rfc_model is None:
        return {pid: None for pid in candidates}

    rows = []
    for pid in candidates:
        prod_rows = df[df['product_id'] == pid]
        usr_rows = df[df['user_id'] == user_id]
        if not prod_rows.empty:
            prod = prod_rows.iloc[0]
        else:
            prod = df.iloc[0]
        if not usr_rows.empty:
            usr = usr_rows.iloc[0]
        else:
            usr = df.iloc[0]
        row = {
            'age': usr.get('age', np.nan),
            'price': prod.get('price', np.nan),
            'rating': prod.get('rating', np.nan),
            'gender': usr.get('gender', 'unknown'),
            'region': usr.get('region', 'unknown'),
            'device': usr.get('device', 'unknown'),
            'category': prod.get('category', 'unknown'),
            'hour': datetime.now().hour,
            'dayofweek': datetime.now().weekday()
        }
        rows.append((pid, row))

    df_rows = pd.DataFrame([r for _, r in rows])
    try:
        feature_names = list(rfc_model.feature_names_in_)
    except Exception:
        feature_names = None

    df_enc = pd.get_dummies(df_rows, columns=['gender','region','device','category','hour','dayofweek'], drop_first=True)

    if feature_names is not None:
        for col in feature_names:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[feature_names]

    df_enc = df_enc.fillna(df_enc.median(numeric_only=True))

    try:
        probs = rfc_model.predict_proba(df_enc)[:,1]
    except Exception as e:
        st.warning("RFC predict_proba failed: " + str(e))
        probs = [None]*len(rows)

    return {pid: float(p) if p is not None else None for (pid, _), p in zip(rows, probs)}

st.sidebar.header("Controls")
user_list = sorted(user_item.index.astype(str).tolist())
selected_user = st.sidebar.selectbox("Select user_id", options=user_list)
num_recs = st.sidebar.number_input("Number of recommendations", min_value=1, max_value=20, value=1, step=1)
show_promotions = st.sidebar.checkbox("Show promotion suggestions", value=True)

predict_btn = st.sidebar.button("Recommend")

st.subheader("Customer profile")
if selected_user:
    sample_profile = data[data['user_id'].astype(str) == str(selected_user)].iloc[0]
    st.write(sample_profile[['user_id','age','gender','region','device']])

if predict_btn:
    with st.spinner("Generating recommendations and scoring..."):
        recs = recommend_for_user(selected_user, n=num_recs*2)
        if not recs:
            st.info("No recommendations available for this user.")
        else:
            recs = recs[:num_recs]
            prod_info = []
            for pid in recs:
                prows = data[data['product_id'] == pid]
                if not prows.empty:
                    p = prows.iloc[0]
                    prod_info.append({'product_id': pid, 'category': p.get('category',''), 'price': p.get('price',np.nan), 'rating': p.get('rating',np.nan)})
                else:
                    prod_info.append({'product_id': pid, 'category': '', 'price': np.nan, 'rating': np.nan})

            scores = score_candidates_with_rfc(selected_user, [r['product_id'] for r in prod_info], rfc, data)
            for r in prod_info:
                r['predicted_click_prob'] = scores.get(r['product_id'])
                if show_promotions:
                    price = r.get('price', np.nan)
                    if np.isnan(price):
                        r['recommended_promo'] = 'View / Bundle'
                    elif price > np.nanpercentile(data['price'].dropna(), 75):
                        r['recommended_promo'] = '15% discount or Free Shipping'
                    elif price > np.nanpercentile(data['price'].dropna(), 50):
                        r['recommended_promo'] = '10% discount'
                    else:
                        r['recommended_promo'] = 'Personalized email + small discount'

            results_df = pd.DataFrame(prod_info)
            results_df = results_df.sort_values(by=['predicted_click_prob'], ascending=False, na_position='last')

            st.subheader("Product Recommendations")
            st.dataframe(results_df.reset_index(drop=True))

            # --- Display targeted ad suggestions (distinct from recommendations) ---
            st.subheader("Targeted Ad Suggestions")

            ad_suggestions = []
            for _, rec in results_df.iterrows():
                cat = rec.get('category', None)
                similar_products = data[(data['category'] == cat) & (data['product_id'] != rec['product_id'])]
                if not similar_products.empty:
                    ad_prod = similar_products.sample(1).iloc[0]
                    ad_suggestions.append({
                        'Product ID': ad_prod['product_id'],
                        'Category': ad_prod['category'],
                        'Price': ad_prod.get('price', np.nan),
                        'Rating': ad_prod.get('rating', np.nan),
                        'Ad Headline': f"Don't miss this {cat} — special offer!",
                        'Ad Copy': f"Recommended for you. {cat} Get it now and save 5% today!",
                        'Suggested Action': 'View Product'
                    })
                else:
                    ad_suggestions.append({
                        'Product ID': f"promo_{rec['product_id']}",
                        'Category': rec.get('category', ''),
                        'Price': np.nan,
                        'Rating': np.nan,
                        'Ad Headline': f"Trending in {rec.get('category','Popular Items')}",
                        'Ad Copy': f"Discover what’s trending among shoppers like you. Limited-time deals on top-rated {rec.get('category','items')}.",
                        'Suggested Action': 'Browse More'
                    })

            ad_df = pd.DataFrame(ad_suggestions)
            st.dataframe(ad_df)

            ads_csv = ad_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download targeted ads CSV", data=ads_csv, file_name=f"ads_user_{selected_user}.csv")

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations CSV", data=csv, file_name=f"recs_user_{selected_user}.csv")

st.info("Note: the app loads saved models from ecom_synthetic/. Run this app in the same folder where you saved the models and data.")

st.sidebar.markdown("---")
