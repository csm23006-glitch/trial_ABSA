import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------------------
# 1. PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LifeCare Bangsar South — ABSA Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 2. CACHED DATA & MODEL LOADING
# ------------------------------------------------------------------------------
@st.cache_data(max_size=10)
def load_absa_data(filepath: str):
    df_absa = pd.read_csv(filepath)
    df_absa["Reviewed_Date"] = pd.to_datetime(df_absa["Reviewed_Date"])
    df_absa["Year"] = df_absa["Reviewed_Date"].dt.year

    # Dynamically recreate Category Summary
    df_cat = (
        df_absa.groupby(["Aspect_Category", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
        .reset_index()
    )
    df_cat["Total_Mentions"] = df_cat[["Negative", "Neutral", "Positive"]].sum(axis=1)

    # Dynamically recreate Top Aspect Terms
    df_terms = (
        df_absa.groupby(["Aspect_Category", "Aspect_Term"])
        .agg(
            Total_Mentions=("Sentiment", "count"),
            Avg_Polarity=("Polarity_Score", "mean")
        )
        .reset_index()
        .sort_values("Total_Mentions", ascending=False)
    )

    # Dynamically recreate Hidden Negatives (4 & 5 Star reviews with Negative aspects)
    df_hidden = df_absa[(df_absa["Rating"].isin([4, 5])) & (df_absa["Sentiment"] == "Negative")].copy()

    return df_absa, df_cat, df_terms, df_hidden

# Call using your CSV filename:
df_absa, df_cat, df_terms, df_hidden = load_absa_data("LIFECARE_ABSA_DATA.csv")


@st.cache_resource
def train_svm_model(df_absa: pd.DataFrame):
    X = df_absa["Aspect_Category"] + " : " + df_absa["Aspect_Term"] + " | " + df_absa["Sentence"]
    y = df_absa["Sentiment"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=7000, sublinear_tf=True, stop_words="english")),
        ("clf", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

df_absa, df_cat, df_terms, df_hidden = load_absa_data("LIFECARE_ABSA_ANALYSIS_OUTPUT.xlsx")
svm_pipeline = train_svm_model(df_absa)

# ------------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# ------------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Patient Feedback")

all_categories = sorted(df_absa["Aspect_Category"].unique())
selected_categories = st.sidebar.multiselect(
    "Operational Categories",
    options=all_categories,
    default=all_categories
)

min_yr, max_yr = int(df_absa["Year"].min()), int(df_absa["Year"].max())
year_range = st.sidebar.slider("Review Year Range", min_yr, max_yr, (2016, max_yr))

selected_ratings = st.sidebar.pills(
    "Google Star Rating",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5],
    selection_mode="multi"
)

filtered_df = df_absa[
    (df_absa["Aspect_Category"].isin(selected_categories)) &
    (df_absa["Year"].between(year_range[0], year_range[1])) &
    (df_absa["Rating"].isin(selected_ratings))
]

# ------------------------------------------------------------------------------
# 4. HEADER & RESPONSIVE KPI METRICS (Streamlit 1.67 st.grid)
# ------------------------------------------------------------------------------
st.title("🏥 LifeCare Diagnostic Medical Centre — ABSA Dashboard")
st.caption("Aspect-Based Sentiment Analysis (ABSA) of Google Reviews (Bangsar South)")

total_mentions = len(filtered_df)
pos_pct = (filtered_df["Sentiment"] == "Positive").mean() * 100 if total_mentions else 0
neu_pct = (filtered_df["Sentiment"] == "Neutral").mean() * 100 if total_mentions else 0
neg_pct = (filtered_df["Sentiment"] == "Negative").mean() * 100 if total_mentions else 0

with st.grid(columns=4):
    st.metric("Extracted Aspect Mentions", f"{total_mentions:,}", border=True)
    st.metric("Positive Aspect Share", f"{pos_pct:.1f}%", border=True)
    st.metric("Neutral Aspect Share", f"{neu_pct:.1f}%", border=True)
    st.metric("Negative Aspect Share", f"{neg_pct:.1f}%", border=True)

# ------------------------------------------------------------------------------
# 5. MAIN DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Operational EDA",
    "🔎 Aspect Term Explorer",
    "⚠️ Hidden Negatives (4–5★)",
    "🤖 Live ABSA Predictor (SVM)"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        cat_grouped = (
            filtered_df.groupby(["Aspect_Category", "Sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig_cat = px.bar(
            cat_grouped,
            x="Count",
            y="Aspect_Category",
            color="Sentiment",
            orientation="h",
            title="Sentiment Breakdown by Operational Category",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        yr_grouped = (
            filtered_df.groupby(["Year", "Sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig_yr = px.bar(
            yr_grouped,
            x="Year",
            y="Count",
            color="Sentiment",
            title="Annual Aspect Sentiment Trend",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    # Heatmap of Star Rating vs Aspect Sentiment
    if not filtered_df.empty:
        rat_ct = pd.crosstab(filtered_df["Rating"], filtered_df["Sentiment"], normalize="index") * 100
        rat_ct = rat_ct.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0).round(1)
        fig_hm = px.imshow(
            rat_ct,
            text_auto=True,
            color_continuous_scale="YlGnBu",
            labels=dict(x="Aspect Sentiment", y="Google Star Rating", color="Row %"),
            title="Star Rating vs. Aspect-Level Sentiment (%)"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

with tab2:
    st.subheader("Top Extracted Aspect Keywords & Polarity")
    cat_pick = st.selectbox("Select Operational Category", ["All"] + all_categories)
    terms_view = df_terms if cat_pick == "All" else df_terms[df_terms["Aspect_Category"] == cat_pick]

    fig_terms = px.bar(
        terms_view.head(15),
        x="Total_Mentions",
        y="Aspect_Term",
        color="Avg_Polarity",
        orientation="h",
        color_continuous_scale="RdYlGn",
        title=f"Top 15 Aspect Keywords ({cat_pick})"
    )
    fig_terms.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_terms, use_container_width=True)
    st.dataframe(terms_view, use_container_width=True)

with tab3:
    st.subheader(f"Hidden Negative Aspects Inside 4★ and 5★ Reviews (n = {len(df_hidden)})")
    st.info(
        "These patients still rated LifeCare 4 or 5 stars overall, but expressed specific operational "
        "friction (e.g., long queues or billing delays) inside their review text."
    )
    st.dataframe(
        df_hidden[["Review_No", "Reviewed_Date", "Rating", "Aspect_Category", "Aspect_Term", "Polarity_Score", "Sentence"]],
        use_container_width=True
    )

with tab4:
    st.subheader("Real-Time Aspect-Based Sentiment Classifier (Linear SVM)")
    with st.form("absa_predict_form"):
        c1, c2 = st.columns(2)
        pred_cat = c1.selectbox("Operational Category", all_categories)
        pred_term = c2.text_input("Target Aspect Term (e.g., waiting time, nurse, blood test)", value="waiting time")
        pred_text = st.text_area(
            "Patient Review Sentence / Clause",
            value="The nurses were very polite and helpful, but the waiting time at the counter took over two hours."
        )
        submitted = st.form_submit_button("Analyze Aspect Sentiment")

    if submitted and pred_text.strip():
        model_query = f"{pred_cat} : {pred_term} | {pred_text}"
        pred_label = svm_pipeline.predict([model_query])[0]
        probs = svm_pipeline.predict_proba([model_query])[0]
        classes = svm_pipeline.classes_

        st.markdown(f"### Predicted Sentiment for **{pred_term}** ({pred_cat}): `{pred_label}`")
        prob_df = pd.DataFrame({"Sentiment": classes, "Confidence": probs})
        fig_prob = px.bar(
            prob_df,
            x="Sentiment",
            y="Confidence",
            color="Sentiment",
            text_auto=".1%",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        st.plotly_chart(fig_prob, use_container_width=True)