from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
DATA_PATH = ROOT / "student_data.csv"

st.set_page_config(
    page_title="EduPredict | Student Progress",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warm editorial palette: parchment, ink, terracotta and muted olive.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root {
  --paper: #f4f0e7; --card: #fffdf8; --ink: #252923;
  --muted: #77796f; --rust: #bd5b3d; --olive: #68745a;
  --line: #ded9cc; --soft: #ebe5d8;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper); color: var(--ink);
  font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(244,240,231,.92); }
[data-testid="stSidebar"] { background: #e9e3d7; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.block-container { max-width: 1220px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { color: var(--ink) !important; }
.hero {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 22px; padding: 30px 34px; margin-bottom: 22px;
  position: relative; overflow: hidden;
}
.hero:after {
  content: ''; position: absolute; width: 190px; height: 190px;
  border: 1px solid #d8b7a8; border-radius: 50%;
  right: -55px; top: -75px; box-shadow: 0 0 0 18px #f7eee6, 0 0 0 19px #e8cfc2;
}
.eyebrow { font: 500 11px 'DM Mono'; letter-spacing: 2px; text-transform: uppercase; color: var(--rust); }
.hero h1 { font: 700 42px 'Playfair Display', serif; margin: 8px 0 5px; letter-spacing: -.7px; }
.hero p { color: var(--muted); max-width: 660px; margin: 0; font-size: 15px; }
.panel {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 18px; padding: 22px 24px; margin-bottom: 16px;
}
.panel-label { font: 500 10px 'DM Mono'; letter-spacing: 1.5px; color: var(--muted); text-transform: uppercase; }
.metric-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 18px 20px; min-height: 108px;
}
.metric-value { font: 700 29px 'Playfair Display'; color: var(--ink); margin-top: 8px; }
.metric-note { color: var(--muted); font-size: 12px; margin-top: 2px; }
.result-good, .result-care {
  border-radius: 16px; padding: 20px 22px; margin-top: 14px;
}
.result-good { background: #e8ede2; border: 1px solid #cbd5c1; }
.result-care { background: #f5e7df; border: 1px solid #e4c5b6; }
.result-title { font: 700 25px 'Playfair Display'; margin: 3px 0; }
.smallcaps { font: 500 10px 'DM Mono'; letter-spacing: 1.3px; text-transform: uppercase; }
div.stButton > button {
  background: var(--rust); color: white; border: 0; border-radius: 999px;
  padding: .62rem 1.25rem; font-weight: 700; min-height: 44px;
}
div.stButton > button:hover { background: #a94e34; color: white; border: 0; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: var(--rust); }
[data-testid="stTabs"] button { color: var(--muted); }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--rust); }
hr { border-color: var(--line); }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}

if not MODEL_PATH.exists():
    st.error("The trained model file is missing.")
    st.code("python3 train.py", language="bash")
    st.stop()

model = load_model()
metrics = load_json(METRICS_PATH)

with st.sidebar:
    st.markdown("<div class='eyebrow'>FIELD STUDY / 01</div>", unsafe_allow_html=True)
    st.markdown("## EduPredict")
    st.caption("A small study of learning signals.")
    st.divider()
    page = st.radio("WORKSPACE", ["Prediction desk", "Model notes", "Dataset"], label_visibility="visible")
    st.divider()
    st.markdown("**Model**")
    st.caption("Decision Tree Classifier")
    st.caption("Artifact · student_model.joblib")
    st.markdown("---")
    st.caption("Learning prototype · synthetic data")

st.markdown("""
<div class="hero">
  <div class="eyebrow">STUDENT OUTCOMES / MACHINE LEARNING</div>
  <h1>Read the signals.<br>Support the journey.</h1>
  <p>An interactive learning project that explores how study time, attendance and previous marks relate to a student's academic outcome.</p>
</div>
""", unsafe_allow_html=True)

if page == "Prediction desk":
    left, right = st.columns([1.03, 1], gap="large")
    with left:
        st.markdown("<div class='panel'><div class='panel-label'>01 / Student profile</div><h3>Set the learning indicators</h3><p style='color:#77796f;font-size:13px'>Adjust the values to explore a sample prediction.</p>", unsafe_allow_html=True)
        study = st.slider("Daily study time", 0.5, 9.0, 4.0, 0.5, format="%.1f hrs")
        attendance = st.slider("Attendance", 45, 100, 80, 1, format="%d%%")
        previous = st.slider("Previous exam marks", 25, 98, 65, 1, format="%d / 100")
        st.markdown("</div>", unsafe_allow_html=True)
        predict = st.button("Generate prediction  →", use_container_width=True)

    with right:
        st.markdown("<div class='panel'><div class='panel-label'>02 / Model output</div><h3>Outcome estimate</h3>", unsafe_allow_html=True)
        sample = pd.DataFrame([{
            "study_hours": study,
            "attendance": attendance,
            "previous_marks": previous
        }])
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        confidence = float(probabilities[prediction])
        if prediction == 1:
            st.markdown(f"""<div class='result-good'><div class='smallcaps' style='color:#68745a'>MODEL INDICATION</div><div class='result-title'>On track</div><div style='color:#52604a'>The model places this profile in the positive outcome group.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='result-care'><div class='smallcaps' style='color:#a94e34'>MODEL INDICATION</div><div class='result-title'>Needs support</div><div style='color:#86523f'>The model places this profile in the support-needed group.</div></div>""", unsafe_allow_html=True)
        st.metric("Model confidence", f"{confidence:.0%}")
        st.progress(confidence)
        st.caption("Confidence is the model's probability estimate, not a guarantee of a student's actual result.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-label'>03 / Input summary</div><h3>Profile at a glance</h3></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, label, value in [
        (c1, "STUDY / DAY", f"{study:.1f} hrs"),
        (c2, "ATTENDANCE", f"{attendance}%"),
        (c3, "PREVIOUS MARKS", f"{previous}/100")
    ]:
        with col:
            st.markdown(f"<div class='metric-card'><div class='panel-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

elif page == "Model notes":
    st.markdown("<div class='panel'><div class='panel-label'>MODEL CARD / 01</div><h2>How the model was built</h2><p style='color:#77796f'>A supervised classification workflow using a held-out test set.</p></div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    vals = [
        ("ACCURACY", metrics.get("accuracy"), "Overall correct predictions"),
        ("PRECISION", metrics.get("precision"), "Positive predictions that were correct"),
        ("RECALL", metrics.get("recall"), "Positive cases identified"),
        ("F1 SCORE", metrics.get("f1_score"), "Balance of precision and recall")
    ]
    for col, (label, val, note) in zip([a,b,c,d], vals):
        with col:
            shown = f"{val:.1%}" if isinstance(val, (int,float)) else "—"
            st.markdown(f"<div class='metric-card'><div class='panel-label'>{label}</div><div class='metric-value'>{shown}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)
    st.write("")
    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown("#### Confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual: Support", "Actual: On track"], columns=["Predicted: Support", "Predicted: On track"]), use_container_width=True)
    st.markdown("#### Workflow")
    st.write("1. Generate and inspect the teaching dataset.")
    st.write("2. Split records into training and testing sets.")
    st.write("3. Fit a Decision Tree Classifier.")
    st.write("4. Evaluate the model on unseen test records.")
    st.write("5. Save the fitted pipeline with Joblib and load it in this app.")

elif page == "Dataset":
    st.markdown("<div class='panel'><div class='panel-label'>DATA / 01</div><h2>Explore the sample records</h2><p style='color:#77796f'>A reproducible synthetic dataset created for this learning exercise.</p></div>", unsafe_allow_html=True)
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.dataframe(data.head(30), use_container_width=True)
        st.download_button("Download dataset CSV", data.to_csv(index=False), "student_data.csv", "text/csv")
        st.caption(f"{len(data)} rows · {len(data.columns)} columns")
    else:
        st.info("Dataset file not found. Run `python3 train.py` to generate it.")

st.markdown("---")
st.caption("EDUPREDICT / EDUCATIONAL PROTOTYPE · Synthetic data only. Do not use this model to make real academic or student-support decisions.")
