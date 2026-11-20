
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="UODA DiaML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

st.markdown(
    """
    <style>
    .diaml-disclaimer {
        margin-top: 2rem;
        padding: 12px 4px 8px;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        color: #94A3B8;
        font-size: 11px;
        line-height: 1.6;
    }

    .diaml-disclaimer-title {
        color: #0D9488;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    @media (prefers-color-scheme: light) {
        .diaml-disclaimer {
            color: #64748B;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

dashboard = st.Page(
    "pages/dashboard.py",
    title="Dashboard",
    icon=":material/dashboard:",
    default=True
)

prediction = st.Page(
    "pages/prediction.py",
    title="Diabetes Prediction",
    icon=":material/monitor_heart:"
)

what_if = st.Page(
    "pages/what_if.py",
    title="What-If Simulator",
    icon=":material/tune:"
)

model_analytics = st.Page(
    "pages/model_analytics.py",
    title="Model Analytics",
    icon=":material/analytics:"
)

batch_prediction = st.Page(
    "pages/batch_prediction.py",
    title="Batch Prediction",
    icon=":material/upload_file:"
)

history = st.Page(
    "pages/history.py",
    title="Prediction History",
    icon=":material/history:"
)

pg = st.navigation(
    [
        dashboard,
        prediction,
        what_if,
        model_analytics,
        batch_prediction,
        history
    ],
    position="hidden"
)

if LOGO_PATH.exists():
    st.logo(
        image=str(LOGO_PATH),
        icon_image=str(LOGO_PATH),
        size="large"
    )

with st.sidebar:
    st.page_link(
        dashboard,
        label="Dashboard",
        icon=":material/dashboard:"
    )

    st.page_link(
        prediction,
        label="Diabetes Prediction",
        icon=":material/monitor_heart:"
    )

    st.page_link(
        what_if,
        label="What-If Simulator",
        icon=":material/tune:"
    )

    st.page_link(
        model_analytics,
        label="Model Analytics",
        icon=":material/analytics:"
    )

    st.page_link(
        batch_prediction,
        label="Batch Prediction",
        icon=":material/upload_file:"
    )

    st.page_link(
        history,
        label="Prediction History",
        icon=":material/history:"
    )

    st.markdown(
        """
        <div class="diaml-disclaimer">
            <div class="diaml-disclaimer-title">
                Educational Use Only
            </div>
            UODA DiaML is an educational project and is not a
            medical diagnostic tool. Predictions must not be
            used for clinical decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

pg.run()
