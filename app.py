"""
Requirements covered: #5 (new-ticket form), #6 (show prediction + confidence + keywords),
#7 (low-confidence warning), plus wiring in the LLM section (#8-#11) and the UI checklist.
Run with: streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils import clean_text
from llm_service import generate_response, refine_response
import config

load_dotenv()

st.set_page_config(page_title="TicketWise", page_icon="🎫", layout="wide")

# Small amount of custom CSS for a cleaner, less "default Streamlit" look.
# Kept intentionally simple: one accent color, card-style containers, nothing fancy.
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1000px; }
    h1 { font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .conf-high { background-color: rgba(46, 160, 67, 0.18); color: #3fb950; }
    .conf-low  { background-color: rgba(219, 68, 55, 0.18); color: #f85149; }
</style>
""", unsafe_allow_html=True)

st.title("🎫 TicketWise")
st.caption("ML ticket classifier + AI-assisted response — for support agents")


@st.cache_resource
def load_artifacts():
    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    return model, vectorizer, metrics


model, vectorizer, metrics = load_artifacts()
best_name = metrics["best_model"]

# Tabs keep the page from turning into one long scroll of unrelated sections.
tab_classify, tab_eval = st.tabs(["📥 Classify a ticket", "📊 Model evaluation"])

# ============================== TAB 1: classify ==============================
with tab_classify:
    # ---------- Section: new ticket form (requirement 5) ----------
    with st.container(border=True):
        st.subheader("New ticket")
        with st.form("ticket_form"):
            subject = st.text_input("Ticket subject")
            description = st.text_area("Ticket description", height=100)
            urgency = st.selectbox("Urgency", ["Low", "Medium", "High", "Critical"])
            submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        if not subject.strip() or not description.strip():
            st.warning("Please fill in both subject and description.")
        else:
            # Same cleaning + vectorizing steps as training, so the model sees
            # the new ticket in the exact same "shape" it learned from
            combined = clean_text(subject + " " + description)
            vec = vectorizer.transform([combined])

            pred_category = model.predict(vec)[0]        # the guessed category
            proba = model.predict_proba(vec)[0]           # probability for every category
            confidence = float(np.max(proba))              # highest probability = how sure it is

            # Requirement 6: figure out WHICH words pushed the model toward this category
            feature_names = np.array(vectorizer.get_feature_names_out())
            nonzero_idx = vec.nonzero()[1]  # only words that actually appear in this ticket
            if hasattr(model, "coef_"):
                class_idx = list(model.classes_).index(pred_category)
                weights = model.coef_[class_idx][nonzero_idx]
            else:
                class_idx = list(model.classes_).index(pred_category)
                weights = model.feature_log_prob_[class_idx][nonzero_idx]
            top_order = np.argsort(weights)[::-1][:5]
            top_words = feature_names[nonzero_idx][top_order]

            st.session_state["last_prediction"] = {
                "subject": subject, "description": description,
                "category": pred_category, "confidence": confidence, "urgency": urgency,
                "top_words": list(top_words),
            }
            st.session_state["last_llm_response"] = None  # reset on new prediction

    # ---------- Section: prediction + confidence + keywords (requirement 6) ----------
    if "last_prediction" in st.session_state:
        p = st.session_state["last_prediction"]
        with st.container(border=True):
            st.subheader("Prediction")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("Predicted category", p["category"])
            with col2:
                badge_class = "conf-high" if p["confidence"] >= config.CONFIDENCE_THRESHOLD else "conf-low"
                st.markdown(
                    f"<br><span class='confidence-badge {badge_class}'>Confidence: {p['confidence']:.0%}</span>",
                    unsafe_allow_html=True,
                )
            st.progress(p["confidence"])
            st.caption("Top influencing words: " + ", ".join(p["top_words"]))

            # ---------- Requirement 7: low-confidence warning ----------
            if p["confidence"] < config.CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Prediction confidence is low. Manual review is recommended.")

    # ---------- Section: LLM-assisted response (requirements 8, 9, 11) ----------
    if "last_prediction" in st.session_state:
        p = st.session_state["last_prediction"]
        with st.container(border=True):
            st.subheader("🤖 AI-assisted response")

            if st.button("Generate assisted response"):
                # Requirement 8: send ticket + ML prediction + urgency to the LLM
                result = generate_response(
                    p["subject"], p["description"], p["category"], p["confidence"], p["urgency"]
                )
                st.session_state["last_llm_response"] = result

            resp = st.session_state.get("last_llm_response")
            if resp:
                if resp["ok"]:
                    # Requirement 9: the three things the LLM hands back
                    st.markdown(f"**Issue summary**\n\n{resp['issue_summary']}")
                    st.markdown(f"**Customer response**\n\n{resp['customer_response']}")
                    st.markdown(f"**Internal action**\n\n{resp['internal_action']}")
                else:
                    # Requirement 11: LLM failed, but the ML prediction above still works fine
                    st.error(f"LLM service unavailable — ML prediction above is still valid. ({resp['error']})")

                # ---------- Requirement 10: refinement follow-up ----------
                if resp["ok"]:
                    st.divider()
                    instruction = st.text_input(
                        "Refine this response (e.g. 'make it shorter', 'more empathetic')"
                    )
                    if st.button("Apply refinement") and instruction.strip():
                        refined = refine_response(resp, instruction)
                        st.session_state["last_llm_response"] = refined
                        st.rerun()

# ============================== TAB 2: evaluation ==============================
with tab_eval:
    st.subheader("Model comparison")
    col1, col2 = st.columns(2)
    for col, name in zip([col1, col2], ["naive_bayes", "logistic_regression"]):
        with col:
            with st.container(border=True):
                r = metrics[name]
                tag = " ⭐" if name == best_name else ""
                st.markdown(f"**{name.replace('_', ' ').title()}{tag}**")
                mcol1, mcol2 = st.columns(2)
                mcol1.metric("Accuracy", f"{r['accuracy']:.2%}")
                mcol2.metric("F1 score", f"{r['f1']:.2%}")
                st.caption(f"Precision {r['precision']:.2%} · Recall {r['recall']:.2%}")

    with st.expander("Confusion matrix (selected model)"):
        best_metrics = metrics[best_name]
        cm_df = pd.DataFrame(
            best_metrics["confusion_matrix"],
            index=best_metrics["labels"],
            columns=best_metrics["labels"],
        )
        st.dataframe(cm_df, use_container_width=True)