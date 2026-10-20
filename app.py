
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_model.pkl"
DATA_FILE = BASE_DIR / "SMSSpamCollection"

st.set_page_config(
    page_title="SpamGuard | Email Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
    data = data.dropna()
    data["label"] = data["label"].str.strip().str.lower()
    data["message"] = data["message"].astype(str)
    return data[data["label"].isin(["ham", "spam"])]


# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(120deg, #102a43, #176b87);
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #e1edf5;
    font-size: 16px;
}
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    min-height: 105px;
}
.metric-label {
    color: #64748b;
    font-size: 14px;
}
.metric-value {
    color: #102a43;
    font-size: 27px;
    font-weight: 700;
    margin-top: 7px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📧 SpamGuard")
    st.caption("Machine Learning Project")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard", "🔍 Detect Message",
         "📊 Model Performance", "ℹ️ About Project"]
    )

    st.divider()
    st.caption("Algorithm: Multinomial Naive Bayes")
    st.caption("Features: TF-IDF")
    st.caption("Framework: Streamlit")

# ---------------- HOME ----------------


if page == "🏠 Dashboard":
    import pickle
    from pathlib import Path

    st.markdown("""
    <div class="hero">
        <h1>🛡️ SpamGuard</h1>
        <p>Intelligent Email Spam Detection Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        base_dir = Path(__file__).resolve().parent

        with open(base_dir / "evaluation.pkl", "rb") as f:
            results = pickle.load(f)

        report = results["report"]

        total = results["train_size"] + results["test_size"]
        spam_count = report["spam"]["support"]
        ham_count = report["ham"]["support"]
        accuracy = results["accuracy"]
        spam_recall = report["spam"]["recall"]
        spam_f1 = report["spam"]["f1-score"]

        st.subheader("Model Overview")

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (c1, "Test Accuracy", f"{accuracy:.1%}"),
            (c2, "Spam Recall", f"{spam_recall:.1%}"),
            (c3, "Spam F1-Score", f"{spam_f1:.1%}"),
            (c4, "Test Emails", f"{results['test_size']:,}")
        ]

        for col, label, value in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.write("")
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Email Dataset")

            distribution = pd.DataFrame({
                "Category": ["Not Spam", "Spam"],
                "Emails": [ham_count, spam_count]
            }).set_index("Category")

            st.bar_chart(distribution)

            st.caption(
                "Class distribution in the held-out test dataset."
            )

        with right:
            st.subheader("How SpamGuard Works")

            st.markdown("""
            **01 — Email Input**

            Enter a message to analyze.

            **02 — Text Processing**

            TF-IDF converts text into numerical features.

            **03 — Machine Learning**

            Multinomial Naive Bayes classifies the message.

            **04 — Prediction**

            View the predicted Spam or Not Spam result.
            """)

        st.write("")
        st.subheader("Technology Stack")

        t1, t2, t3 = st.columns(3)

        t1.info("**Python**\n\nProgramming language")
        t2.info("**Scikit-learn**\n\nTF-IDF + Naive Bayes")
        t3.info("**Streamlit**\n\nInteractive web app")

        st.caption(
            f"Training emails: {results['train_size']:,} · "
            f"Test emails: {results['test_size']:,} · "
            "Academic prototype; predictions are not a security guarantee."
        )

    except FileNotFoundError:
        st.error(
            "Evaluation file not found. Run python train_model.py first."
        )
    except Exception as e:
        st.error(f"Could not load model statistics: {e}")

# ---------------- PREDICTION ----------------

elif page == "🔍 Detect Message":
    st.markdown("""
    <div class="hero">
        <h1>🔍 Email & Message Analyzer</h1>
        <p>Analyze a message using the SpamGuard machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Test a Message")
    st.write(
        "Choose a sample message or paste your own text below."
    )

    samples = {
        "Choose a sample...": "",
        "Spam example": (
            "Congratulations! You have won a free prize. "
            "Click now to claim your reward!"
        ),
        "Normal example": (
            "Hi, are we meeting at the college library tomorrow "
            "to complete our assignment?"
        ),
        "Suspicious offer": (
            "WIN a free ticket now! Call immediately to claim "
            "your exclusive cash reward."
        ),
        "College announcement": (
            "Hello everyone, the project review will be held "
            "in the classroom tomorrow at 10 AM."
        )
    }

    choice = st.selectbox(
        "Choose a sample",
        list(samples.keys())
    )

    message = st.text_area(
        "Message content",
        value=samples[choice],
        height=200,
        placeholder="Paste the email subject and message body here..."
    )

    st.caption(
        "Tip: Including both the subject and message body may provide "
        "useful context for the classifier."
    )

    if st.button(
        "🔎 Analyze Message",
        type="primary",
        use_container_width=True
    ):
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
        else:
            try:
                model = load_model()
                prediction = model.predict([message])[0]

                st.divider()
                st.subheader("Analysis Result")

                if prediction == "spam":
                    st.markdown("""
                    <div style="
                        padding: 22px;
                        border-radius: 14px;
                        background: #fff1f2;
                        border: 1px solid #fecdd3;
                        color: #9f1239;
                    ">
                        <h2 style="margin:0;">🚨 SPAM DETECTED</h2>
                        <p style="margin:8px 0 0 0;">
                            The model classified this message as spam.
                            Review the sender and links carefully.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == "ham":
                    st.markdown("""
                    <div style="
                        padding: 22px;
                        border-radius: 14px;
                        background: #f0fdf4;
                        border: 1px solid #bbf7d0;
                        color: #166534;
                    ">
                        <h2 style="margin:0;">✅ NOT SPAM</h2>
                        <p style="margin:8px 0 0 0;">
                            The model classified this message as
                            not spam.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(
                        f"The model returned an unexpected label: {prediction}"
                    )

                st.caption(
                    "This is an automated prediction, not a guarantee "
                    "that a message is safe or harmful. Avoid clicking "
                    "unexpected links or sharing sensitive information."
                )

            except FileNotFoundError:
                st.error(
                    "Trained model not found. Run train_model.py first."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------- MODEL PERFORMANCE ----------------

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write("Evaluation on a held-out real email test dataset.")

    try:
        import pickle
        from pathlib import Path

        evaluation_path = (
            Path(__file__).resolve().parent / "evaluation.pkl"
        )

        with open(evaluation_path, "rb") as f:
            results = pickle.load(f)

        accuracy = results["accuracy"]
        report = results["report"]
        cm = results["confusion_matrix"]

        spam_precision = report["spam"]["precision"]
        spam_recall = report["spam"]["recall"]
        spam_f1 = report["spam"]["f1-score"]

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (c1, "Accuracy", accuracy),
            (c2, "Spam Precision", spam_precision),
            (c3, "Spam Recall", spam_recall),
            (c4, "Spam F1-Score", spam_f1)
        ]

        for col, label, value in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.caption(
            f"Training emails: {results['train_size']} | "
            f"Test emails: {results['test_size']}"
        )

        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Not Spam", "Actual Spam"],
            columns=["Predicted Not Spam", "Predicted Spam"]
        )

        st.dataframe(cm_df, use_container_width=True)

        st.caption(
            "Rows represent actual labels; columns represent predicted labels."
        )

        st.subheader("Classification Report")

        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.rename(
            index={
                "ham": "Not Spam",
                "spam": "Spam"
            }
        )

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    except FileNotFoundError:
        st.error(
            "evaluation.pkl not found. Run python train_model.py first."
        )
    except Exception as e:
        st.error(f"Could not load evaluation results: {e}")



# ---------------- ABOUT ----------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About the Project")

    st.markdown("""
    ### Spam Email Detection Using Machine Learning

    **Objective:** Classify email messages into Spam and Not Spam
    categories using machine learning.

    **Technologies Used**
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Multinomial Naive Bayes
    - Streamlit

    **Dataset**
    - Apache SpamAssassin Public Corpus
    - Ham (legitimate) emails and spam emails
    - Total emails: 2,996
    - Ham emails: 2,502
    - Spam emails: 494

    **Methodology**
    1. Load email files and their contents.
    2. Extract email subjects and message text.
    3. Convert text into numerical features using TF-IDF.
    4. Train a Multinomial Naive Bayes classifier.
    5. Evaluate the model using a held-out test dataset.
    6. Predict whether a new email is spam or not spam.

    **Model Evaluation**
    The model is evaluated using accuracy, precision, recall,
    F1-score, and a confusion matrix.

    **Project Type:** Academic machine learning prototype.

    **Limitations**
    - Predictions are not guaranteed to be correct.
    - Performance may vary on emails not represented in the training data.
    - Suspicious links and attachments should always be checked carefully.
    """)

    st.warning(
        "A Not Spam prediction does not guarantee that an email is safe."
    )
