import hashlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IONS = ["Na+", "Li+", "Mg2+", "Ca2+", "K+", "Cl-", "NO3-", "SO4 2-"]
MODELS = ["CatBoost", "XGBoost", "Random Forest", "LightGBM"]


@dataclass
class PredictionResult:
    percent: float
    bars: pd.DataFrame


def _stable_float(seed: str, lower: float, upper: float) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    ratio = int(digest[:8], 16) / 0xFFFFFFFF
    return lower + ratio * (upper - lower)


def build_mock_prediction(
    organic_concentration: float,
    ph_value: float,
    ions: list[str],
    ion_concentration: float,
    model_name: str,
) -> PredictionResult:
    shown_ions = ions[:3] or ["Na+", "Li+", "Mg2+"]
    base = f"{organic_concentration:.4f}|{ph_value:.4f}|{ion_concentration:.4f}|{','.join(shown_ions)}|{model_name}"
    percent = round(_stable_float(base, -30.0, 20.0), 1)

    rows = []
    for index, ion in enumerate(shown_ions):
        value_seed = f"{base}|{ion}|{index}"
        value = round(_stable_float(value_seed, -3.4, 3.2), 1)
        rows.append(
            {
                "ion": ion,
                "change": value,
                "color": "#055CFF" if value >= 0 else "#12C6CF",
            }
        )

    return PredictionResult(percent=percent, bars=pd.DataFrame(rows))


def default_prediction() -> PredictionResult:
    return PredictionResult(
        percent=-12.5,
        bars=pd.DataFrame(
            [
                {"ion": "Na+", "change": 1.8, "color": "#055CFF"},
                {"ion": "Li+", "change": -2.6, "color": "#12C6CF"},
                {"ion": "Mg2+", "change": -0.9, "color": "#6753F5"},
            ]
        ),
    )


def parse_float(raw_value: str, field_name: str) -> float | None:
    value = raw_value.strip()
    if not value:
        st.error(f"Please enter {field_name}.")
        return None

    try:
        return float(value)
    except ValueError:
        st.error(f"{field_name} must be a number.")
        return None


def draw_bar_chart(result: PredictionResult) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.35), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        result.bars["ion"],
        result.bars["change"],
        color=result.bars["color"],
        width=0.44,
        edgecolor="none",
    )

    ax.axhline(0, color="#8895AA", linewidth=1)
    ax.set_ylim(-4, 4)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_ylabel("Change (mg/g)", fontsize=10)
    ax.set_title("Equilibrium Adsorption Capacity Change", fontsize=12, pad=10, fontweight="bold")
    ax.grid(axis="y", color="#EBF0F8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9C4D6")
    ax.spines["bottom"].set_color("#B9C4D6")
    ax.tick_params(axis="x", labelsize=10, colors="#1B1E2D")
    ax.tick_params(axis="y", labelsize=9, colors="#34405A")

    for bar, value, color in zip(bars, result.bars["change"], result.bars["color"]):
        x = bar.get_x() + bar.get_width() / 2
        offset = 0.18 if value >= 0 else -0.18
        va = "bottom" if value >= 0 else "top"
        sign = "+" if value > 0 else ""
        ax.text(
            x,
            value + offset,
            f"{sign}{value:.1f}",
            ha="center",
            va=va,
            fontsize=10,
            color=color,
        )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"], [data-testid="stHeader"], footer {
                display: none;
            }

            .stApp {
                background: linear-gradient(180deg, #FFFFFF 0%, #F5F8FF 100%);
            }

            .block-container {
                max-width: 1220px;
                padding: 34px 34px 18px;
            }

            .app-title {
                margin: 0 0 28px;
                text-align: center;
                color: #045DFF !important;
                font-size: 34px;
                font-weight: 800;
                letter-spacing: 0;
            }

            .section-title {
                color: #045DFF;
                font-size: 24px;
                font-weight: 800;
                margin: 2px 0 22px;
            }

            .result-label {
                color: #1C2536;
                font-size: 17px;
                text-align: center;
                margin-top: 0;
                margin-bottom: 12px;
            }

            .result-value {
                color: #064FE6;
                font-size: 74px;
                line-height: 1;
                font-weight: 800;
                text-align: center;
                margin: 0 0 34px;
            }

            .hint-box {
                border: 1px solid #DFE8F7;
                border-radius: 8px;
                background: #FBFDFF;
                color: #435278;
                padding: 12px 14px;
                margin-bottom: 12px;
                font-size: 14px;
                line-height: 1.65;
            }

            .ai-badge {
                border: 1px solid #DDE8FA;
                border-radius: 8px;
                padding: 16px 12px;
                margin: 8px 0;
                text-align: center;
                color: #045DFF;
                font-size: 30px;
                font-weight: 800;
                min-height: 74px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: #DDE8FA;
                border-radius: 8px;
                box-shadow: 0 12px 30px rgba(16, 70, 140, 0.04);
                background: rgba(255, 255, 255, 0.94);
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stFormSubmitButton"] > button {
                border-radius: 7px;
                border: 1px solid #055CFF;
                min-height: 46px;
                font-size: 17px;
                font-weight: 600;
            }

            div[data-testid="stFormSubmitButton"] > button {
                background: #055CFF;
                color: white;
            }

            div[data-testid="stButton"] > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                border-color: #0046C7;
                color: #0046C7;
            }

            div[data-testid="stFormSubmitButton"] > button:hover {
                color: white;
                background: #004FE0;
            }

            div[role="radiogroup"] {
                gap: 12px;
            }

            div[role="radiogroup"] label {
                border: 1px solid #DDE8FA;
                border-radius: 8px;
                padding: 10px 16px;
                min-height: 48px;
                background: #FFFFFF;
            }

            div[role="radiogroup"] label:has(input:checked) {
                border-color: #055CFF;
                box-shadow: 0 0 0 1px #055CFF inset;
                color: #055CFF;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding: 24px 18px 14px;
                }

                .app-title {
                    font-size: 26px;
                    margin-bottom: 20px;
                }

                .result-value {
                    font-size: 54px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("model_name", "CatBoost")
    st.session_state.setdefault("last_result", default_prediction())
    st.session_state.setdefault("has_predicted", False)
    st.session_state.setdefault("show_metrics", False)


def render_input_card() -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Parameter Input</div>', unsafe_allow_html=True)

        with st.form("prediction_form", clear_on_submit=False):
            organic_raw = st.text_input(
                "Initial Organic Concentration",
                placeholder="Enter concentration",
                label_visibility="visible",
            )
            st.caption("Unit: mg/L")

            ph_raw = st.text_input("pH", placeholder="Enter pH")

            selected_ions = st.multiselect(
                "Ion Type",
                options=IONS,
                default=[],
                placeholder="Select ion types (multiple allowed)",
            )
            st.caption("Note: Select metal ions that do not precipitate with the organic compound.")

            ion_concentration_raw = st.text_input("Ion Concentration", placeholder="Enter concentration")
            st.caption("Unit: mol/L")

            submitted = st.form_submit_button("Start Prediction", width="stretch")

        if submitted:
            organic = parse_float(organic_raw, "initial organic concentration")
            ph_value = parse_float(ph_raw, "pH")
            ion_concentration = parse_float(ion_concentration_raw, "ion concentration")

            if not selected_ions:
                st.error("Please select at least one ion type.")
                return

            if organic is None or ph_value is None or ion_concentration is None:
                return

            if organic <= 0:
                st.error("Initial organic concentration must be greater than 0.")
                return

            if ion_concentration <= 0:
                st.error("Ion concentration must be greater than 0.")
                return

            if not 0 <= ph_value <= 14:
                st.error("pH must be between 0 and 14.")
                return

            st.session_state.last_result = build_mock_prediction(
                organic,
                ph_value,
                selected_ions,
                ion_concentration,
                st.session_state.model_name,
            )
            st.session_state.has_predicted = True
            st.success("Prediction complete. Mock results are shown.")


def render_result_card() -> None:
    result = st.session_state.last_result
    sign = "+" if result.percent > 0 else ""

    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="result-label">Relative Adsorption Capacity Change (vs. Blank)</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-value">{sign}{result.percent:.1f}%</div>',
            unsafe_allow_html=True,
        )

        draw_bar_chart(result)
        st.markdown(
            """
            <div class="hint-box">
                Vs. blank: change in equilibrium adsorption capacity after adding ions compared with the blank system.<br>
                Up to 3 ion types are displayed.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_metrics() -> None:
    metrics = {
        "CatBoost": ("0.93", "0.87", "0.18"),
        "XGBoost": ("0.91", "0.84", "0.22"),
        "Random Forest": ("0.88", "0.79", "0.27"),
        "LightGBM": ("0.90", "0.82", "0.24"),
    }
    r2, rmse, mae = metrics[st.session_state.model_name]

    with st.container(border=True):
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        col_r2, col_rmse, col_mae = st.columns(3)
        col_r2.metric("R²", r2)
        col_rmse.metric("RMSE", rmse)
        col_mae.metric("MAE", mae)
        st.caption("Mock performance metrics. Replace them with validation results after the real models are connected.")


def render_bottom_bar() -> None:
    with st.container(border=True):
        badge_col, model_col, current_col, button_col = st.columns([0.8, 6.4, 1.8, 1.8], vertical_alignment="center")
        with badge_col:
            st.markdown('<div class="ai-badge">AI</div>', unsafe_allow_html=True)

        with model_col:
            selected_model = st.radio(
                "Model Selection",
                MODELS,
                index=MODELS.index(st.session_state.model_name),
                horizontal=True,
                label_visibility="collapsed",
            )
            st.session_state.model_name = selected_model

        with current_col:
            st.markdown(
                f"Current Model: <span style='color:#045DFF;font-weight:700'>{st.session_state.model_name}</span>",
                unsafe_allow_html=True,
            )

        with button_col:
            if st.button("Model Performance", width="stretch"):
                st.session_state.show_metrics = not st.session_state.show_metrics

    if st.session_state.show_metrics:
        render_model_metrics()


def main() -> None:
    st.set_page_config(
        page_title="Relative Adsorption Capacity Prediction System",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_state()

    st.markdown('<h1 class="app-title">Relative Adsorption Capacity Prediction System</h1>', unsafe_allow_html=True)

    input_col, result_col = st.columns([0.47, 0.53], gap="large")
    with input_col:
        render_input_card()
    with result_col:
        render_result_card()

    st.write("")
    render_bottom_bar()


if __name__ == "__main__":
    main()
