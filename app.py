import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. PAGE CONFIG
st.set_page_config(
    page_title="LifeCare Bangsar South — ABSA Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

# 2. CACHED DATA & MODELS LOADING
@st.cache_data(max_size=10)
def load_absa_data(filepath: str):
    raw_df = pd.read_csv(filepath)
    raw_df["Reviewed Date"] = pd.to_datetime(raw_df["Reviewed Date"])
    raw_df["Year"] = raw_df["Reviewed Date"].dt.year

    # Explode pipe-separated aspect columns into aspect-level rows
    rows = []
    for _, r in raw_df.iterrows():
        if pd.isna(r["ASPECT_TERMS"]) or str(r["ASPECT_TERMS"]).strip() == "":
            continue
        terms = [t.strip() for t in str(r["ASPECT_TERMS"]).split("|")]
        cats = [c.strip() for c in str(r["ASPECT_CATEGORIES"]).split("|")]
        sents = [s.strip() for s in str(r["ASPECT_SENTIMENTS"]).split("|")]
        sentence_txt = r["REVIEW_ENGLISH"] if pd.notna(r["REVIEW_ENGLISH"]) else r["Review"]

        for t, c, s in zip(terms, cats, sents):
            rows.append({
                "Review_No": r["No"],
                "Reviewer_Name": r["Reviewer's name"],
                "Reviewed_Date": r["Reviewed Date"],
                "Year": r["Year"],
                "Rating": r["Rating"],
                "Sentence": sentence_txt,
                "Aspect_Term": t,
                "Aspect_Category": c,
                "Sentiment": s
            })

    df_absa = pd.DataFrame(rows)

    # Top Aspect Terms
    polarity_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df_absa["Polarity_Num"] = df_absa["Sentiment"].map(polarity_map).fillna(0)
    df_terms = (
        df_absa.groupby(["Aspect_Category", "Aspect_Term"])
        .agg(
            Total_Mentions=("Sentiment", "count"),
            Positive_Mentions=("Sentiment", lambda x: (x == "Positive").sum()),
            Negative_Mentions=("Sentiment", lambda x: (x == "Negative").sum()),
            Net_Sentiment_Score=("Polarity_Num", "mean")
        )
        .reset_index()
        .sort_values("Total_Mentions", ascending=False)
    )

    # Hidden Negatives (4 & 5 Star reviews with Negative aspects)
    df_hidden = df_absa[(df_absa["Rating"].isin([4, 5])) & (df_absa["Sentiment"] == "Negative")].copy()

    return raw_df, df_absa, df_terms, df_hidden

@st.cache_resource
def train_all_models(df_absa: pd.DataFrame):
    X = df_absa["Aspect_Category"] + " : " + df_absa["Aspect_Term"] + " | " + df_absa["Sentence"]
    y = df_absa["Sentiment"]

    classifiers = {
        "SVM (Linear, Balanced — Best: 84.23% Acc)": SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42),
        "Naive Bayes (Multinomial — 80.53% Acc)": MultinomialNB(alpha=0.5),
        "KNN (k=5, Distance-Weighted — 78.11% Acc)": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    }

    trained_pipes = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=7000, sublinear_tf=True, stop_words="english")),
            ("clf", clf)
        ])
        pipe.fit(X, y)
        trained_pipes[name] = pipe

    return trained_pipes

raw_df, df_absa, df_terms, df_hidden = load_absa_data("LIFECARE_ABSA_INDEPENDENT_RESULTS.csv")
models_dict = train_all_models(df_absa)

# 3. SIDEBAR FILTERS
st.sidebar.header("🔍 Filter Patient Feedback")

all_categories = sorted(df_absa["Aspect_Category"].unique())
selected_categories = st.sidebar.multiselect(
    "Operational Categories (ABSA)",
    options=all_categories,
    default=all_categories
)

min_yr, max_yr = int(raw_df["Year"].min()), int(raw_df["Year"].max())
year_range = st.sidebar.slider("Review Year Range", min_yr, max_yr, (2016, max_yr))

selected_ratings = st.sidebar.pills(
    "Google Star Rating",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5],
    selection_mode="multi"
)

filtered_raw = raw_df[
    (raw_df["Year"].between(year_range[0], year_range[1])) &
    (raw_df["Rating"].isin(selected_ratings))
]

filtered_absa = df_absa[
    (df_absa["Aspect_Category"].isin(selected_categories)) &
    (df_absa["Year"].between(year_range[0], year_range[1])) &
    (df_absa["Rating"].isin(selected_ratings))
]

# 4. HEADER & KPI METRICS
st.title("🏥 LifeCare Diagnostic Medical Centre — ABSA Dashboard")
st.caption("Pre-ABSA Star Rating Overview & 7-Category Clause-Level Aspect-Based Sentiment Analysis (Bangsar South)")

total_reviews = len(filtered_raw)
avg_rating = filtered_raw["Rating"].mean() if total_reviews else 0
total_mentions = len(filtered_absa)
pos_pct = (filtered_absa["Sentiment"] == "Positive").mean() * 100 if total_mentions else 0
neg_pct = (filtered_absa["Sentiment"] == "Negative").mean() * 100 if total_mentions else 0

with st.grid(columns=5):
    st.metric("Total Reviews (Pre-ABSA)", f"{total_reviews:,}", border=True)
    st.metric("Average Star Rating", f"{avg_rating:.2f} ★", border=True)
    st.metric("Extracted Aspect Mentions", f"{total_mentions:,}", border=True)
    st.metric("Positive Aspect Share", f"{pos_pct:.1f}%", border=True)
    st.metric("Negative Aspect Share", f"{neg_pct:.1f}%", border=True)

# 5. MAIN DASHBOARD TABS
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "⭐ Pre-ABSA Star Rating EDA",
    "📊 Post-ABSA Operational EDA",
    "🔎 Aspect Term Explorer",
    "⚠️ Hidden Negatives (4–5★)",
    "🤖 Live ABSA Predictor (SVM / NB / KNN)"
])

with tab0:
    st.subheader("General Review & Star Rating Distribution (Before ABSA Splitting)")
    c_left, c_right = st.columns(2)

    with c_left:
        star_counts = (
            filtered_raw["Rating"]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .reset_index()
        )
        star_counts.columns = ["Star Rating", "Review Count"]
        star_counts["Percentage"] = (star_counts["Review Count"] / max(total_reviews, 1) * 100).round(1)
        star_counts["Label"] = star_counts.apply(lambda r: f"{r['Review Count']:,} ({r['Percentage']}%)", axis=1)

        fig_stars = px.bar(
            star_counts,
            x="Star Rating",
            y="Review Count",
            text="Label",
            color="Star Rating",
            color_continuous_scale=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
            title="Google Review Count by Star Rating (1★ to 5★)"
        )
        fig_stars.update_traces(textposition="outside")
        fig_stars.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig_stars, use_container_width=True)

    with c_right:
        yr_rev = (
            filtered_raw.groupby(["Year", "Rating"])
            .size()
            .reset_index(name="Review Count")
        )
        yr_rev["Rating"] = yr_rev["Rating"].astype(str) + " ★"
        fig_yr_rev = px.bar(
            yr_rev,
            x="Year",
            y="Review Count",
            color="Rating",
            title="Annual Review Volume by Star Rating",
            color_discrete_map={
                "1 ★": "#e74c3c",
                "2 ★": "#e67e22",
                "3 ★": "#f1c40f",
                "4 ★": "#58d68d",
                "5 ★": "#27ae60"
            }
        )
        st.plotly_chart(fig_yr_rev, use_container_width=True)

    st.dataframe(star_counts[["Star Rating", "Review Count", "Percentage"]], use_container_width=True)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        cat_grouped = (
            filtered_absa.groupby(["Aspect_Category", "Sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig_cat = px.bar(
            cat_grouped,
            x="Count",
            y="Aspect_Category",
            color="Sentiment",
            orientation="h",
            title="Sentiment Breakdown Across 7 Operational Categories",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        yr_grouped = (
            filtered_absa.groupby(["Year", "Sentiment"])
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

    if not filtered_absa.empty:
        rat_ct = pd.crosstab(filtered_absa["Rating"], filtered_absa["Sentiment"], normalize="index") * 100
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
    st.subheader("Top Extracted Aspect Keywords & Net Sentiment")
    cat_pick = st.selectbox("Select Operational Category", ["All"] + all_categories)
    terms_view = df_terms if cat_pick == "All" else df_terms[df_terms["Aspect_Category"] == cat_pick]

    fig_terms = px.bar(
        terms_view.head(15),
        x="Total_Mentions",
        y="Aspect_Term",
        color="Net_Sentiment_Score",
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
        df_hidden[["Review_No", "Reviewed_Date", "Rating", "Aspect_Category", "Aspect_Term", "Sentence"]],
        use_container_width=True
    )

with tab4:
    st.subheader("Real-Time Aspect-Based Sentiment Classifier")
    with st.form("absa_predict_form"):
        selected_model_name = st.selectbox("Choose Classifier Model", list(models_dict.keys()))
        c1, c2 = st.columns(2)
        pred_cat = c1.selectbox("Operational Category", all_categories)
        pred_term = c2.text_input("Target Aspect Term (e.g., waiting time, nurse, blood test)", value="waiting time")
        pred_text = st.text_area(
            "Patient Review Sentence / Clause",
            value="The nurses were very polite and helpful, but the waiting time at the counter took over two hours."
        )
        submitted = st.form_submit_button("Analyze Aspect Sentiment")

    if submitted and pred_text.strip():
        active_pipe = models_dict[selected_model_name]
        model_query = f"{pred_cat} : {pred_term} | {pred_text}"
        pred_label = active_pipe.predict([model_query])[0]
        probs = active_pipe.predict_proba([model_query])[0]
        classes = active_pipe.classes_

        st.markdown(f"### Predicted Sentiment ({selected_model_name.split(' —')[0]}): `{pred_label}`")
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
