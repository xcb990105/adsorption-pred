from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from descriptors import (
    BLANK_CAPACITY,
    DescriptorError,
    available_anion_types,
    available_ion_types,
    build_feature_frame,
    load_descriptor_map,
    relative_change_percent,
)
from modeling import load_metrics, load_model, model_names, predict_capacity


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODELS_DIR = "models"
TRAINING_CONCENTRATION_RANGE = (0.0, 0.5)
TRAINING_PH_RANGE = (3.0, 11.0)


@dataclass
class PredictionResult:
    capacity: float
    percent: float
    bars: pd.DataFrame


@st.cache_resource
def cached_descriptor_map() -> dict:
    return load_descriptor_map(MODELS_DIR)


@st.cache_resource
def cached_model(model_name: str):
    return load_model(model_name, MODELS_DIR)


@st.cache_data
def cached_metrics() -> dict:
    return load_metrics(MODELS_DIR)


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

    values = result.bars["change"]
    y_min = min(-10.0, values.min() - 5.0)
    y_max = max(10.0, values.max() + 5.0)

    bars = ax.bar(
        result.bars["ion"],
        values,
        color=result.bars["color"],
        width=0.44,
        edgecolor="none",
    )

    ax.axhline(0, color="#8895AA", linewidth=1)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Change vs. Blank (%)", fontsize=10)
    ax.set_title("Relative Adsorption Capacity Change", fontsize=12, pad=10, fontweight="bold")
    ax.grid(axis="y", color="#EBF0F8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9C4D6")
    ax.spines["bottom"].set_color("#B9C4D6")
    ax.tick_params(axis="x", labelsize=10, colors="#1B1E2D")
    ax.tick_params(axis="y", labelsize=9, colors="#34405A")

    for bar, value, color in zip(bars, values, result.bars["color"]):
        x = bar.get_x() + bar.get_width() / 2
        offset = 0.9 if value >= 0 else -0.9
        va = "bottom" if value >= 0 else "top"
        sign = "+" if value > 0 else ""
        ax.text(
            x,
            value + offset,
            f"{sign}{value:.1f}%",
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
                font-size: 66px;
                line-height: 1;
                font-weight: 800;
                text-align: center;
                margin: 0 0 10px;
            }

            .result-subvalue {
                color: #064FE6;
                font-size: 36px;
                line-height: 1.15;
                font-weight: 750;
                text-align: center;
                margin: 0 0 30px;
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
                    font-size: 48px;
                }

                .result-subvalue {
                    font-size: 28px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    names = model_names()
    st.session_state.setdefault("model_name", "CatBoost")
    if st.session_state.model_name not in names:
        st.session_state.model_name = names[0]
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("has_predicted", False)
    st.session_state.setdefault("show_metrics", False)


def warn_outside_training_range(concentration: float, ph_value: float) -> None:
    c_min, c_max = TRAINING_CONCENTRATION_RANGE
    ph_min, ph_max = TRAINING_PH_RANGE
    if not c_min <= concentration <= c_max:
        st.warning(f"Ion concentration is outside the training range ({c_min:g}-{c_max:g} mol/L).")
    if not ph_min <= ph_value <= ph_max:
        st.warning(f"pH is outside the training range ({ph_min:g}-{ph_max:g}).")


def make_prediction_result(
    model_name: str,
    descriptor_map: dict,
    salt_anion_type: str,
    selected_ions: list[str],
    concentration: float,
    ph_value: float,
) -> PredictionResult:
    model = cached_model(model_name)
    rows = []

    for ion in selected_ions:
        feature_frame = build_feature_frame(
            descriptor_map,
            salt_anion_type,
            ion,
            concentration,
            ph_value,
        )
        capacity = predict_capacity(model, feature_frame)
        change = relative_change_percent(capacity)
        rows.append(
            {
                "ion": ion,
                "capacity": capacity,
                "change": change,
                "color": "#055CFF" if change >= 0 else "#12C6CF",
            }
        )

    result_table = pd.DataFrame(rows)
    primary = result_table.iloc[0]

    return PredictionResult(
        capacity=float(primary["capacity"]),
        percent=float(primary["change"]),
        bars=result_table.head(3).copy(),
    )


def render_input_card(descriptor_map: dict) -> None:
    anion_options = available_anion_types(descriptor_map)

    with st.container(border=True):
        st.markdown('<div class="section-title">Parameter Input</div>', unsafe_allow_html=True)

        st.markdown("Salt Anion Type")
        salt_anion_type = st.selectbox(
            "Salt Anion Type",
            options=anion_options,
            label_visibility="collapsed",
        )
        ion_options = available_ion_types(descriptor_map, salt_anion_type)

        with st.form("prediction_form", clear_on_submit=False):
            st.markdown("Ion Type")
            selected_ions = st.multiselect(
                "Ion Type",
                options=ion_options,
                default=[],
                placeholder="Select ion types (multiple allowed)",
                label_visibility="collapsed",
            )
            st.caption("Available ions are limited to descriptor mappings included with the deployed model.")

            st.markdown("Ion Concentration")
            ion_concentration_raw = st.text_input(
                "Ion Concentration",
                placeholder="Enter concentration",
                label_visibility="collapsed",
            )
            st.caption("Unit: mol/L")

            st.markdown("pH")
            ph_raw = st.text_input("pH", placeholder="Enter pH", label_visibility="collapsed")

            submitted = st.form_submit_button("Start Prediction", width="stretch")

        if submitted:
            ion_concentration = parse_float(ion_concentration_raw, "ion concentration")
            ph_value = parse_float(ph_raw, "pH")

            if not selected_ions:
                st.error("Please select at least one ion type.")
                return

            if ion_concentration is None or ph_value is None:
                return

            if ion_concentration < 0:
                st.error("Ion concentration must be 0 or greater.")
                return

            if not 0 <= ph_value <= 14:
                st.error("pH must be between 0 and 14.")
                return

            warn_outside_training_range(ion_concentration, ph_value)

            try:
                st.session_state.last_result = make_prediction_result(
                    st.session_state.model_name,
                    descriptor_map,
                    salt_anion_type,
                    selected_ions,
                    ion_concentration,
                    ph_value,
                )
            except (DescriptorError, FileNotFoundError) as exc:
                st.error(str(exc))
                return

            st.session_state.has_predicted = True
            st.success("Prediction complete.")


def render_result_card() -> None:
    result = st.session_state.last_result

    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="result-label">Adsorption Capacity</p>',
            unsafe_allow_html=True,
        )

        if result is None:
            st.markdown('<div class="result-value">--</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-subvalue">Enter parameters to run a model prediction.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="hint-box">
                    The deployed app loads trained model artifacts from the models directory and does not read raw data.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        sign = "+" if result.percent > 0 else ""
        st.markdown(
            f'<div class="result-value">{result.capacity:.2f} mg/g</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-subvalue">{sign}{result.percent:.1f}% vs. blank</div>',
            unsafe_allow_html=True,
        )

        draw_bar_chart(result)
        st.markdown(
            f"""
            <div class="hint-box">
                Blank capacity: {BLANK_CAPACITY:.4f} mg/g.<br>
                Relative change = (predicted capacity - blank capacity) / blank capacity * 100%.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_metrics() -> None:
    metrics = cached_metrics()
    model_metrics = metrics.get(st.session_state.model_name, {})
    r2_value = model_metrics.get("cv_r2_mean", model_metrics.get("test_r2", float("nan")))
    rmse_value = model_metrics.get("cv_rmse_mean", model_metrics.get("test_rmse", float("nan")))
    mae_value = model_metrics.get("cv_mae_mean", model_metrics.get("test_mae", float("nan")))

    with st.container(border=True):
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        col_r2, col_rmse, col_mae = st.columns(3)
        col_r2.metric("CV R2", f"{r2_value:.3f}")
        col_rmse.metric("CV RMSE", f"{rmse_value:.3f}")
        col_mae.metric("CV MAE", f"{mae_value:.3f}")


def render_bottom_bar() -> None:
    with st.container(border=True):
        badge_col, model_col, current_col, button_col = st.columns([0.8, 6.4, 1.8, 1.8], vertical_alignment="center")
        with badge_col:
            st.markdown('<div class="ai-badge">AI</div>', unsafe_allow_html=True)

        with model_col:
            names = model_names()
            selected_model = st.radio(
                "Model Selection",
                names,
                index=names.index(st.session_state.model_name),
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

    try:
        descriptor_map = cached_descriptor_map()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    input_col, result_col = st.columns([0.47, 0.53], gap="large")
    with input_col:
        render_input_card(descriptor_map)
    with result_col:
        render_result_card()

    st.write("")
    render_bottom_bar()


if __name__ == "__main__":
    main()
