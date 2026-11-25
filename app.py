import re
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
    page_title="LifeCare Bangsar South — End-to-End ABSA Dashboard",
    page_icon="🏥",
    layout="wide"
)

# 2. RULE-BASED 7-CATEGORY CLAUSE-LEVEL ABSA ENGINE (FOR RAW CSVs)
CATEGORY_LEXICON = {
    "Staff & Concierge": ["staff", "nurse", "nurses", "reception", "receptionist", "front desk", "counter", "concierge", "security", "guard", "supervisor", "attendant", "helpful", "rude", "friendly", "polite"],
    "Queue & Turnaround Time": ["wait", "waiting", "queue", "queued", "slow", "fast", "quick", "delay", "delayed", "hours", "hour", "time", "turnaround", "long", "process", "appointment", "punctual"],
    "Medical & Visa Screening": ["medical", "checkup", "check-up", "check up", "screening", "visa", "emgs", "fomema", "immigration", "health", "package", "annual"],
    "Diagnostics & Reporting": ["blood", "blood test", "x-ray", "xray", "ultrasound", "scan", "mri", "ct", "ecg", "lab", "laboratory", "report", "reports", "result", "results", "urine", "sample"],
    "Facility & Digital Access": ["facility", "facilities", "lounge", "toilet", "washroom", "clean", "dirty", "parking", "building", "aircon", "cold", "wifi", "app", "website", "online", "system", "place", "area"],
    "Clinical & Doctors": ["doctor", "doctors", "doc", "dr", "physician", "specialist", "consultation", "consult", "examine", "examination", "diagnosis", "treatment", "advice", "explain"],
    "Billing & Insurance": ["bill", "billing", "price", "cost", "charge", "charged", "fee", "fees", "expensive", "cheap", "insurance", "panel", "gl", "guarantee letter", "claim", "payment", "pay", "refund"]
}

POS_WORDS = {"good", "great", "excellent", "friendly", "helpful", "polite", "fast", "quick", "smooth", "clean", "comfortable", "comfy", "satisfied", "professional", "kind", "efficient", "breeze", "best", "nice", "courteous", "responsive", "attentive", "amazing"}
NEG_WORDS = {"bad", "poor", "slow", "rude", "unhelpful", "long", "delay", "delayed", "dirty", "expensive", "terrible", "worst", "horrible", "waste", "upset", "frustrated", "frustration", "disappointed", "crowded", "confusing", "careless", "unprofessional", "late", "overcharged", "cold"}
NEGATORS = {"not", "no", "never", "neither", "hardly", "barely", "didnt", "didn't", "wasnt", "wasn't", "isnt", "isn't", "dont", "don't"}

def score_clause_sentiment(clause: str) -> str:
    tokens = re.findall(r"[a-z']+", clause.lower())
    score = 0
    for i, tok in enumerate(tokens):
        negated = any(prev in NEGATORS for prev in tokens[max(0, i - 3):i])
        if tok in POS_WORDS:
            score += -1 if negated else 1
        elif tok in NEG_WORDS:
            score += 1 if negated else -1
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"

def run_raw_absa_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Detect review text, date, rating, and ID columns flexibly
    text_col = next((c for c in ["REVIEW_ENGLISH", "Review", "review_text", "Text"] if c in raw_df.columns), None)
    date_col = next((c for c in ["Reviewed Date", "Reviewed_Date", "Date", "date"] if c in raw_df.columns), None)
    rating_col = next((c for c in ["Rating", "rating", "Stars"] if c in raw_df.columns), None)
    id_col = next((c for c in ["No", "Review_No", "ID"] if c in raw_df.columns), None)

    if text_col is None or rating_col is None:
        st.error("Raw CSV must contain at least a Review text column and a Rating column.")
        st.stop()

    df = raw_df.copy()
    if id_col is None:
        df["No"] = range(1, len(df) + 1)
        id_col = "No"
    if date_col is not None:
        df["Reviewed_Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["Year"] = df["Reviewed_Date"].dt.year.fillna(2025).astype(int)
    else:
        df["Reviewed_Date"] = pd.NaT
        df["Year"] = 2025

    rows = []
    for _, r in df.iterrows():
        txt = str(r[text_col]) if pd.notna(r[text_col]) else ""
        if not txt.strip():
            continue
        # Clause-level contrast splitting
        clauses = re.split(r"[.!?;]+|\b(?:but|however|although|though|yet|whereas|except)\b", txt, flags=re.IGNORECASE)
        for clause in clauses:
            c_clean = clause.strip()
            if len(c_clean) < 3:
                continue
            c_low = c_clean.lower()
            sent = score_clause_sentiment(c_clean)
            for cat, keywords in CATEGORY_LEXICON.items():
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", c_low):
                        rows.append({
                            "Review_No": r[id_col],
                            "Reviewed_Date": r["Reviewed_Date"],
                            "Year": int(r["Year"]),
                            "Rating": int(r[rating_col]),
                            "Sentence": c_clean,
                            "Aspect_Term": kw,
                            "Aspect_Category": cat,
                            "Sentiment": sent
                        })
                        break  # One match per category per clause
    return pd.DataFrame(rows)

@st.cache_data(max_size=10)
def load_or_process_data(raw_df: pd.DataFrame):
    # If dataset already has pre-computed ABSA columns, explode them directly
    if {"ASPECT_TERMS", "ASPECT_CATEGORIES", "ASPECT_SENTIMENTS"}.issubset(raw_df.columns):
        raw_df = raw_df.copy()
        raw_df["Reviewed_Date"] = pd.to_datetime(raw_df["Reviewed Date"], errors="coerce")
        raw_df["Year"] = raw_df["Reviewed_Date"].dt.year.fillna(2025).astype(int)
        rows = []
        for _, r in raw_df.iterrows():
            if pd.isna(r["ASPECT_TERMS"]) or str(r["ASPECT_TERMS"]).strip() == "":
                continue
            terms = [t.strip() for t in str(r["ASPECT_TERMS"]).split("|")]
            cats = [c.strip() for c in str(r["ASPECT_CATEGORIES"]).split("|")]
            sents = [s.strip() for s in str(r["ASPECT_SENTIMENTS"]).split("|")]
            sentence_txt = r["REVIEW_ENGLISH"] if "REVIEW_ENGLISH" in raw_df.columns and pd.notna(r["REVIEW_ENGLISH"]) else r["Review"]
            for t, c, s in zip(terms, cats, sents):
                rows.append({
                    "Review_No": r["No"],
                    "Reviewed_Date": r["Reviewed_Date"],
                    "Year": int(r["Year"]),
                    "Rating": int(r["Rating"]),
                    "Sentence": sentence_txt,
                    "Aspect_Term": t,
                    "Aspect_Category": c,
                    "Sentiment": s
                })
        df_absa = pd.DataFrame(rows)
        mode_label = "Pre-Computed ABSA Results Loaded"
    else:
        # Raw dataset with NO ABSA columns -> run live clause-level ABSA pipeline
        if "Reviewed Date" in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df["Reviewed_Date"] = pd.to_datetime(raw_df["Reviewed Date"], errors="coerce")
            raw_df["Year"] = raw_df["Reviewed_Date"].dt.year.fillna(2025).astype(int)
        else:
            raw_df["Year"] = 2025
        df_absa = run_raw_absa_pipeline(raw_df)
        mode_label = "Live Clause-Level ABSA Executed on Raw Dataset"

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
    df_hidden = df_absa[(df_absa["Rating"].isin([4, 5])) & (df_absa["Sentiment"] == "Negative")].copy()
    return raw_df, df_absa, df_terms, df_hidden, mode_label

@st.cache_resource
def train_all_models(df_absa: pd.DataFrame):
    X = df_absa["Aspect_Category"] + " : " + df_absa["Aspect_Term"] + " | " + df_absa["Sentence"]
    y = df_absa["Sentiment"]
    classifiers = {
        "SVM (Linear, Balanced — Best: 84.23% Acc)": SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42),
        "Naive Bayes (Multinomial — 80.53% Acc)": MultinomialNB(alpha=0.5),
        "KNN (k=5, Distance-Weighted — 78.11% Acc)": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    }
    trained = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=7000, sublinear_tf=True, stop_words="english")),
            ("clf", clf)
        ])
        pipe.fit(X, y)
        trained[name] = pipe
    return trained

# 3. FILE SOURCE (UPLOAD RAW CSV OR USE DEFAULT REPO CSV)
st.sidebar.header("📂 Dataset Source")
uploaded_file = st.sidebar.file_uploader("Upload Raw or Processed CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    # Change default filename here if you upload a raw CSV to GitHub instead
    input_df = pd.read_csv("LIFECARE_ABSA_INDEPENDENT_RESULTS.csv")

raw_df, df_absa, df_terms, df_hidden, mode_label = load_or_process_data(input_df)
models_dict = train_all_models(df_absa)

# 4. SIDEBAR FILTERS
st.sidebar.header("🔍 Filter Patient Feedback")
all_categories = sorted(df_absa["Aspect_Category"].unique())
selected_categories = st.sidebar.multiselect("Operational Categories (ABSA)", all_categories, default=all_categories)

min_yr, max_yr = int(raw_df["Year"].min()), int(raw_df["Year"].max())
year_range = st.sidebar.slider("Review Year Range", min_yr, max_yr, (min_yr, max_yr)) if min_yr < max_yr else (min_yr, max_yr)

selected_ratings = st.sidebar.pills("Google Star Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], selection_mode="multi")

filtered_raw = raw_df[(raw_df["Year"].between(year_range[0], year_range[1])) & (raw_df["Rating"].isin(selected_ratings))]
filtered_absa = df_absa[
    (df_absa["Aspect_Category"].isin(selected_categories)) &
    (df_absa["Year"].between(year_range[0], year_range[1])) &
    (df_absa["Rating"].isin(selected_ratings))
]

# 5. DYNAMIC HEADER & KPI METRICS
if "Place" in raw_df.columns and raw_df["Place"].dropna().nunique() > 0:
    facility_name = str(raw_df["Place"].dropna().mode()[0]).strip()
elif uploaded_file is not None:
    facility_name = uploaded_file.name.replace(".csv", "").replace("_", " ").title()
else:
    facility_name = "LifeCare Diagnostic Medical Centre — Bangsar South"

st.title(f"🏥 {facility_name} — ABSA Dashboard")
st.caption(f"Mode: **{mode_label}** | Pre-ABSA Star Rating & 7-Category Clause-Level ABSA")
# 6. DASHBOARD TABS
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
        star_counts = filtered_raw["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).reset_index()
        star_counts.columns = ["Star Rating", "Review Count"]
        star_counts["Percentage"] = (star_counts["Review Count"] / max(total_reviews, 1) * 100).round(1)
        star_counts["Label"] = star_counts.apply(lambda r: f"{r['Review Count']:,} ({r['Percentage']}%)", axis=1)
        fig_stars = px.bar(
            star_counts, x="Star Rating", y="Review Count", text="Label", color="Star Rating",
            color_continuous_scale=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
            title="Google Review Count by Star Rating (1★ to 5★)"
        )
        fig_stars.update_traces(textposition="outside")
        fig_stars.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig_stars, use_container_width=True)
    with c_right:
        yr_rev = filtered_raw.groupby(["Year", "Rating"]).size().reset_index(name="Review Count")
        yr_rev["Rating"] = yr_rev["Rating"].astype(str) + " ★"
        fig_yr_rev = px.bar(
            yr_rev, x="Year", y="Review Count", color="Rating", title="Annual Review Volume by Star Rating",
            color_discrete_map={"1 ★": "#e74c3c", "2 ★": "#e67e22", "3 ★": "#f1c40f", "4 ★": "#58d68d", "5 ★": "#27ae60"}
        )
        st.plotly_chart(fig_yr_rev, use_container_width=True)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        cat_grouped = filtered_absa.groupby(["Aspect_Category", "Sentiment"]).size().reset_index(name="Count")
        fig_cat = px.bar(
            cat_grouped, x="Count", y="Aspect_Category", color="Sentiment", orientation="h",
            title="Sentiment Breakdown Across 7 Operational Categories",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)
    with col2:
        yr_grouped = filtered_absa.groupby(["Year", "Sentiment"]).size().reset_index(name="Count")
        fig_yr = px.bar(
            yr_grouped, x="Year", y="Count", color="Sentiment", title="Annual Aspect Sentiment Trend",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        st.plotly_chart(fig_yr, use_container_width=True)

with tab2:
    st.subheader("Top Extracted Aspect Keywords & Net Sentiment")
    cat_pick = st.selectbox("Select Operational Category", ["All"] + all_categories)
    terms_view = df_terms if cat_pick == "All" else df_terms[df_terms["Aspect_Category"] == cat_pick]
    fig_terms = px.bar(
        terms_view.head(15), x="Total_Mentions", y="Aspect_Term", color="Net_Sentiment_Score",
        orientation="h", color_continuous_scale="RdYlGn", title=f"Top 15 Aspect Keywords ({cat_pick})"
    )
    fig_terms.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_terms, use_container_width=True)
    st.dataframe(terms_view, use_container_width=True)

with tab3:
    st.subheader(f"Hidden Negative Aspects Inside 4★ and 5★ Reviews (n = {len(df_hidden)})")
    st.dataframe(df_hidden[["Review_No", "Reviewed_Date", "Rating", "Aspect_Category", "Aspect_Term", "Sentence"]], use_container_width=True)

with tab4:
    st.subheader("Real-Time Aspect-Based Sentiment Classifier")
    with st.form("absa_predict_form"):
        selected_model_name = st.selectbox("Choose Classifier Model", list(models_dict.keys()))
        c1, c2 = st.columns(2)
        pred_cat = c1.selectbox("Operational Category", all_categories)
        pred_term = c2.text_input("Target Aspect Term", value="waiting time")
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
        st.markdown(f"### Predicted Sentiment ({selected_model_name.split(' —')[0]}): `{pred_label}`")
        prob_df = pd.DataFrame({"Sentiment": active_pipe.classes_, "Confidence": probs})
        fig_prob = px.bar(
            prob_df, x="Sentiment", y="Confidence", color="Sentiment", text_auto=".1%",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
        )
        st.plotly_chart(fig_prob, use_container_width=True)
