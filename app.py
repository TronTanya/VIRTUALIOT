"""VirtualIoT — интерфейс Streamlit.

Запуск: streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from controller import DEVICE_NAMES, ConfigError, DeviceStates, Thresholds
from reporting import (
    build_html_report,
    config_to_json,
    events_for_export,
    frame_for_export,
    summary_table,
    to_csv_bytes,
)
from scenarios import (
    AUTOMATED_LABEL,
    BASELINE_LABEL,
    PRESETS,
    ComparisonResult,
    HysteresisExperiment,
    preset_config,
    run_comparison,
    run_hysteresis_experiment,
)
from simulation import (
    MAX_DURATION_MIN,
    MIN_DURATION_MIN,
    MODEL_VERSION,
    TIME_STEP_MIN,
    DevicePower,
    ModelCoefficients,
    SimulationConfig,
    SimulationResult,
    TargetRanges,
    run_simulation,
)

ACCENT = "#315EFB"
SUCCESS = "#278451"
WARNING = "#B7791F"
TEXT = "#202733"
MUTED = "#667085"
BORDER = "#E4E7EC"
GRID = "#EEF0F3"

st.set_page_config(page_title="VirtualIoT", layout="wide")
st.markdown(
    "<style>[data-testid='stMainBlockContainer']{max-width:1100px;padding-top:2rem}</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- состояние

def widget_values(cfg: SimulationConfig) -> dict[str, Any]:
    th, tg, c, p, m = cfg.thresholds, cfg.targets, cfg.coefficients, cfg.power, cfg.manual_states
    return {
        "init_temp": float(cfg.initial_temperature),
        "init_hum": float(cfg.initial_humidity),
        "out_temp": float(cfg.outdoor_temperature),
        "out_hum": float(cfg.outdoor_humidity),
        "nat_lux": float(cfg.natural_illuminance),
        "nat_var_pct": float(cfg.natural_variation * 100.0),
        "duration": int(cfg.duration_min),
        "seed": int(cfg.seed),
        "noise_enabled": bool(cfg.noise_enabled),
        "automation_enabled": bool(cfg.automation_enabled),
        "th_heater_on": float(th.heater_on_below),
        "th_heater_off": float(th.heater_off_above),
        "th_fan_off": float(th.fan_off_below),
        "th_fan_on": float(th.fan_on_above),
        "th_lamp_on": float(th.lamp_on_below),
        "th_lamp_off": float(th.lamp_off_above),
        "man_heater": bool(m.heater),
        "man_fan": bool(m.fan),
        "man_lamp": bool(m.lamp),
        "tg_temp_min": float(tg.temp_min),
        "tg_temp_max": float(tg.temp_max),
        "tg_hum_min": float(tg.hum_min),
        "tg_hum_max": float(tg.hum_max),
        "tg_lux_min": float(tg.lux_min),
        "tg_lux_max": float(tg.lux_max),
        "c_k_temp": float(c.k_temp),
        "c_k_temp_vent": float(c.k_temp_vent),
        "c_heater_gain": float(c.heater_gain),
        "c_k_hum": float(c.k_hum),
        "c_k_hum_vent": float(c.k_hum_vent),
        "c_lamp_lux": float(c.lamp_lux),
        "c_period": float(c.natural_period_min),
        "c_noise_temp": float(c.noise_temp),
        "c_noise_hum": float(c.noise_hum),
        "c_noise_lux": float(c.noise_lux),
        "p_heater": float(p.heater_w),
        "p_fan": float(p.fan_w),
        "p_lamp": float(p.lamp_w),
    }


def apply_config_to_widgets(cfg: SimulationConfig) -> None:
    for key, value in widget_values(cfg).items():
        st.session_state[key] = value


def init_state() -> None:
    if "initialized" not in st.session_state:
        apply_config_to_widgets(SimulationConfig())
        st.session_state.initialized = True
    # Streamlit удаляет значения неотрисованных виджетов (например, порогов в ручном режиме).
    for key in widget_values(SimulationConfig()):
        st.session_state[key] = st.session_state[key]


def config_from_widgets() -> SimulationConfig:
    s = st.session_state
    return SimulationConfig(
        initial_temperature=float(s.init_temp),
        initial_humidity=float(s.init_hum),
        outdoor_temperature=float(s.out_temp),
        outdoor_humidity=float(s.out_hum),
        natural_illuminance=float(s.nat_lux),
        natural_variation=float(s.nat_var_pct) / 100.0,
        duration_min=int(s.duration),
        seed=int(s.seed),
        noise_enabled=bool(s.noise_enabled),
        automation_enabled=bool(s.automation_enabled),
        thresholds=Thresholds(
            float(s.th_heater_on), float(s.th_heater_off),
            float(s.th_fan_off), float(s.th_fan_on),
            float(s.th_lamp_on), float(s.th_lamp_off),
        ),
        manual_states=DeviceStates(bool(s.man_heater), bool(s.man_fan), bool(s.man_lamp)),
        targets=TargetRanges(
            float(s.tg_temp_min), float(s.tg_temp_max),
            float(s.tg_hum_min), float(s.tg_hum_max),
            float(s.tg_lux_min), float(s.tg_lux_max),
        ),
        coefficients=ModelCoefficients(
            float(s.c_k_temp), float(s.c_k_temp_vent), float(s.c_heater_gain),
            float(s.c_k_hum), float(s.c_k_hum_vent), float(s.c_lamp_lux),
            float(s.c_period), float(s.c_noise_temp), float(s.c_noise_hum), float(s.c_noise_lux),
        ),
        power=DevicePower(float(s.p_heater), float(s.p_fan), float(s.p_lamp)),
    )


def load_preset(select_key: str) -> None:
    apply_config_to_widgets(preset_config(st.session_state[select_key]))


def reset_defaults() -> None:
    apply_config_to_widgets(SimulationConfig())


# ---------------------------------------------------------------- форматирование

def num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def state_html(on: bool) -> str:
    color, label = (SUCCESS, "Включён") if on else (MUTED, "Выключен")
    return f"<span style='color:{color};font-weight:600'>{label}</span>"


def base_layout(fig: go.Figure, title: str, y_title: str, height: int = 280) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": TEXT}, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 70, "r": 16, "t": 48, "b": 52},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": TEXT, "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"size": 11}},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Время моделирования, мин", showgrid=False, linecolor=BORDER, ticks="outside", tickcolor=BORDER)
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER)
    return fig


def add_band(fig: go.Figure, lo: float, hi: float) -> None:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=SUCCESS, opacity=0.08, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker={"size": 10, "color": "rgba(39,132,81,0.25)", "symbol": "square"},
                             name="Целевой диапазон"))


def add_thresholds(fig: go.Figure, lo: float, hi: float) -> None:
    for value in (lo, hi):
        fig.add_hline(y=value, line={"color": WARNING, "width": 1, "dash": "dot"})
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line={"color": WARNING, "width": 1, "dash": "dot"}, name="Пороги гистерезиса"))


def sensor_chart(result: SimulationResult, quantity: str) -> go.Figure:
    cfg, df = result.config, result.frame
    tg, th = cfg.targets, cfg.thresholds
    fig = go.Figure()
    if quantity == "temperature":
        add_band(fig, tg.temp_min, tg.temp_max)
        if cfg.automation_enabled:
            add_thresholds(fig, th.heater_on_below, th.heater_off_above)
        fig.add_trace(go.Scatter(x=df["t"], y=df["temp"], name="Датчик", line={"color": ACCENT, "width": 1.6}))
        return base_layout(fig, "Температура (смоделированные данные)", "Температура, °C")
    if quantity == "humidity":
        add_band(fig, tg.hum_min, tg.hum_max)
        if cfg.automation_enabled:
            add_thresholds(fig, th.fan_off_below, th.fan_on_above)
        fig.add_trace(go.Scatter(x=df["t"], y=df["hum"], name="Датчик", line={"color": ACCENT, "width": 1.6}))
        return base_layout(fig, "Относительная влажность (смоделированные данные)", "Влажность, %")
    add_band(fig, tg.lux_min, tg.lux_max)
    if cfg.automation_enabled:
        add_thresholds(fig, th.lamp_on_below, th.lamp_off_above)
    fig.add_trace(go.Scatter(x=df["t"], y=df["lux"], name="Освещённость помещения", line={"color": ACCENT, "width": 1.6}))
    fig.add_trace(go.Scatter(x=df["t"], y=df["lux_ambient"], name="Естественный свет (датчик для лампы)",
                             line={"color": MUTED, "width": 1.2, "dash": "dash"}))
    return base_layout(fig, "Освещённость (смоделированные данные)", "Освещённость, лк")


def device_chart(result: SimulationResult) -> go.Figure:
    df = result.frame
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[DEVICE_NAMES[d] for d in ("heater", "fan", "lamp")])
    for i, device in enumerate(("heater", "fan", "lamp"), start=1):
        values = df[device].astype("Float64").ffill().astype(float)
        fig.add_trace(go.Scatter(x=df["t"], y=values, line_shape="hv", showlegend=False,
                                 line={"color": SUCCESS, "width": 1.6}, fill="tozeroy",
                                 fillcolor="rgba(39,132,81,0.10)", name=DEVICE_NAMES[device]), row=i, col=1)
        fig.update_yaxes(tickvals=[0, 1], ticktext=["Выкл.", "Вкл."], range=[-0.15, 1.25],
                         showgrid=False, row=i, col=1)
    base_layout(fig, "", "", height=300)
    fig.update_layout(margin={"l": 70, "r": 16, "t": 28, "b": 52}, hovermode=False)
    fig.update_xaxes(title_text="", showgrid=False)
    fig.update_xaxes(title_text="Время моделирования, мин", row=3, col=1)
    fig.update_yaxes(title_text="")
    fig.update_annotations(font={"size": 12, "color": MUTED}, x=0, xanchor="left")
    return fig


def comparison_chart(comp: ComparisonResult, quantity: str) -> go.Figure:
    column, title, unit = {
        "Температура": ("temp", "Температура", "Температура, °C"),
        "Влажность": ("hum", "Относительная влажность", "Влажность, %"),
        "Освещённость": ("lux", "Освещённость помещения", "Освещённость, лк"),
    }[quantity]
    tg = comp.automated.config.targets
    band = {"temp": (tg.temp_min, tg.temp_max), "hum": (tg.hum_min, tg.hum_max), "lux": (tg.lux_min, tg.lux_max)}[column]
    fig = go.Figure()
    add_band(fig, *band)
    fig.add_trace(go.Scatter(x=comp.baseline.frame["t"], y=comp.baseline.frame[column], name=BASELINE_LABEL,
                             line={"color": MUTED, "width": 1.4}))
    fig.add_trace(go.Scatter(x=comp.automated.frame["t"], y=comp.automated.frame[column], name=AUTOMATED_LABEL,
                             line={"color": ACCENT, "width": 1.6}))
    return base_layout(fig, f"{title}: сравнение сценариев (смоделированные данные)", unit, height=320)


def hysteresis_chart(exp: HysteresisExperiment) -> go.Figure:
    column, unit = {"heater": ("temp", "Температура, °C"), "fan": ("hum", "Влажность, %"),
                    "lamp": ("lux_ambient", "Естественный свет, лк")}[exp.device]
    fig = go.Figure()
    for res, label, color in ((exp.narrow, "Узкая полоса", WARNING), (exp.wide, "Широкая полоса", ACCENT)):
        fig.add_trace(go.Scatter(x=res.frame["t"], y=res.frame[column], name=label, line={"color": color, "width": 1.4}))
    return base_layout(fig, "Показания датчика при разной ширине гистерезиса", unit, height=300)


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "displayModeBar": False})


# ---------------------------------------------------------------- вкладка 1

def render_preset_picker(prefix: str) -> None:
    names = list(PRESETS)
    c1, c2, c3 = st.columns([3, 1.2, 1.2], vertical_alignment="bottom")
    c1.selectbox("Пресет параметров", names, key=f"{prefix}_preset")
    c2.button("Загрузить пресет", key=f"{prefix}_load", on_click=load_preset, args=(f"{prefix}_preset",), width="stretch")
    c3.button("Сбросить параметры", key=f"{prefix}_reset", on_click=reset_defaults, width="stretch")
    st.caption(PRESETS[st.session_state[f"{prefix}_preset"]]["description"])


def render_inputs() -> None:
    with st.container(border=True):
        st.markdown("**1. Начальные условия**")
        c = st.columns(4)
        c[0].number_input("Температура в помещении, °C", -20.0, 50.0, step=0.5, key="init_temp")
        c[1].number_input("Влажность в помещении, %", 0.0, 100.0, step=1.0, key="init_hum")
        c[2].number_input("Температура на улице, °C", -40.0, 50.0, step=0.5, key="out_temp")
        c[3].number_input("Влажность на улице, %", 0.0, 100.0, step=1.0, key="out_hum")
        c = st.columns(4)
        c[0].number_input("Естественный свет, лк", 0.0, 100_000.0, step=10.0, key="nat_lux")
        c[1].number_input("Колебания естественного света, %", 0.0, 100.0, step=5.0, key="nat_var_pct",
                          help="Амплитуда синусоидальных колебаний вокруг среднего значения (облачность).")
        c[2].number_input(f"Длительность, мин ({MIN_DURATION_MIN}–{MAX_DURATION_MIN})",
                          MIN_DURATION_MIN, MAX_DURATION_MIN, step=10, key="duration")
        c[3].number_input("Зерно генератора шума", 0, 2**32 - 1, step=1, key="seed",
                          help="Одинаковое зерно даёт одинаковый шум датчиков и воспроизводимый результат.")
        st.checkbox("Шум датчиков", key="noise_enabled",
                    help="Шум влияет только на показания датчиков, но не на состояние среды.")

    with st.container(border=True):
        st.markdown("**2. Автоматизация**")
        st.toggle("Автоматизация включена", key="automation_enabled")
        if st.session_state.automation_enabled:
            c = st.columns(3)
            with c[0]:
                st.caption("Обогреватель (по температуре)")
                st.number_input("Включить ниже, °C", -20.0, 50.0, step=0.1, key="th_heater_on")
                st.number_input("Выключить выше, °C", -20.0, 50.0, step=0.1, key="th_heater_off")
            with c[1]:
                st.caption("Вентиляция (по влажности)")
                st.number_input("Включить выше, %", 0.0, 100.0, step=1.0, key="th_fan_on")
                st.number_input("Выключить ниже, %", 0.0, 100.0, step=1.0, key="th_fan_off")
            with c[2]:
                st.caption("Лампа (по естественному свету)")
                st.number_input("Включить ниже, лк", 0.0, 100_000.0, step=10.0, key="th_lamp_on")
                st.number_input("Выключить выше, лк", 0.0, 100_000.0, step=10.0, key="th_lamp_off")
            st.caption(
                "Между порогами устройство сохраняет текущее состояние (гистерезис). "
                "Значение, равное порогу, считается внутри полосы. Лампа управляется по отдельному "
                "датчику естественного света, который не видит свет самой лампы."
            )
        else:
            st.caption("Ручной режим: выбранные состояния действуют весь прогон.")
            c = st.columns(3)
            c[0].checkbox("Обогреватель включён", key="man_heater")
            c[1].checkbox("Вентиляция включена", key="man_fan")
            c[2].checkbox("Лампа включена", key="man_lamp")

    with st.container(border=True):
        st.markdown("**3. Целевые диапазоны** — используются для оценки условий, а не для управления")
        c = st.columns(6)
        c[0].number_input("Темп. мин, °C", -20.0, 50.0, step=0.5, key="tg_temp_min")
        c[1].number_input("Темп. макс, °C", -20.0, 50.0, step=0.5, key="tg_temp_max")
        c[2].number_input("Влажн. мин, %", 0.0, 100.0, step=1.0, key="tg_hum_min")
        c[3].number_input("Влажн. макс, %", 0.0, 100.0, step=1.0, key="tg_hum_max")
        c[4].number_input("Освещ. мин, лк", 0.0, 100_000.0, step=10.0, key="tg_lux_min")
        c[5].number_input("Освещ. макс, лк", 0.0, 100_000.0, step=10.0, key="tg_lux_max")

    with st.expander("Коэффициенты модели и мощность устройств"):
        c = st.columns(5)
        c[0].number_input("k_T, 1/мин", 0.0001, 0.9999, step=0.001, format="%.4f", key="c_k_temp",
                          help="Обмен теплом с улицей без вентиляции.")
        c[1].number_input("k_TV, 1/мин", 0.0, 0.9999, step=0.005, format="%.4f", key="c_k_temp_vent",
                          help="Добавка к теплообмену при работе вентиляции.")
        c[2].number_input("Нагрев h, °C/мин", 0.0, 2.0, step=0.05, key="c_heater_gain")
        c[3].number_input("k_H, 1/мин", 0.0001, 0.9999, step=0.001, format="%.4f", key="c_k_hum")
        c[4].number_input("k_HV, 1/мин", 0.0, 0.9999, step=0.005, format="%.4f", key="c_k_hum_vent")
        c = st.columns(5)
        c[0].number_input("Вклад лампы L, лк", 0.0, 5000.0, step=10.0, key="c_lamp_lux")
        c[1].number_input("Период света P, мин", 10.0, 1440.0, step=10.0, key="c_period")
        c[2].number_input("Шум темп., °C (СКО)", 0.0, 2.0, step=0.05, key="c_noise_temp")
        c[3].number_input("Шум влажн., % (СКО)", 0.0, 10.0, step=0.1, key="c_noise_hum")
        c[4].number_input("Шум освещ., лк (СКО)", 0.0, 200.0, step=1.0, key="c_noise_lux")
        c = st.columns(5)
        c[0].number_input("Обогреватель, Вт", 0.0, 5000.0, step=50.0, key="p_heater")
        c[1].number_input("Вентиляция, Вт", 0.0, 5000.0, step=5.0, key="p_fan")
        c[2].number_input("Лампа, Вт", 0.0, 5000.0, step=5.0, key="p_lamp")
        st.caption("Суммы k_T + k_TV и k_H + k_HV не должны превышать 1, иначе расчёт за один шаг "
                   "перескакивает через уличное значение.")


def render_results(result: SimulationResult, stale: bool, comparison: ComparisonResult | None) -> None:
    cfg = result.config
    if stale:
        st.warning("Параметры изменены после последнего запуска. Ниже — результаты для прежних параметров; "
                   "нажмите «Запустить симуляцию» для пересчёта.")
    st.subheader("Результаты моделирования")
    mode = "автоматизация включена" if cfg.automation_enabled else "ручной режим"
    st.caption(f"Смоделированные данные · конечное наблюдение t = {cfg.duration_min} мин · {mode} · "
               f"зерно {cfg.seed} · шаг {TIME_STEP_MIN} мин · модель {MODEL_VERSION}")

    fr = result.final_readings
    with st.container(border=True):
        c = st.columns(4)
        c[0].metric("Температура", f"{num(fr.temperature)} °C")
        c[1].metric("Влажность", f"{num(fr.humidity)} %")
        c[2].metric("Освещённость", f"{num(fr.illuminance_total, 0)} лк")
        c[3].metric("Естественный свет", f"{num(fr.illuminance_ambient, 0)} лк",
                    help="Отдельный датчик для управления лампой: не учитывает свет самой лампы.")

    with st.container(border=True):
        c = st.columns(3)
        for col, device in zip(c, ("heater", "fan", "lamp")):
            m = result.device_metrics[device]
            col.markdown(f"**{DEVICE_NAMES[device]}** · {state_html(result.final_states.as_dict()[device])}",
                         unsafe_allow_html=True)
            col.caption(f"Работал {m.enabled_minutes} из {cfg.duration_min} мин · переключений: {m.switches} · "
                        f"≈ {num(m.energy_kwh, 3)} кВт·ч")
        st.caption("Состояние — на последнем интервале моделирования.")

    with st.container(border=True):
        tg = cfg.targets
        c = st.columns(4)
        c[0].metric(f"Температура в {num(tg.temp_min)}…{num(tg.temp_max)} °C",
                    f"{num(result.in_range_pct['temperature'])} %")
        c[1].metric(f"Влажность в {num(tg.hum_min, 0)}…{num(tg.hum_max, 0)} %",
                    f"{num(result.in_range_pct['humidity'])} %")
        c[2].metric(f"Освещённость в {num(tg.lux_min, 0)}…{num(tg.lux_max, 0)} лк",
                    f"{num(result.in_range_pct['illuminance'])} %")
        c[3].metric("Электроэнергия (оценка)", f"{num(result.total_energy_kwh, 3)} кВт·ч")
        st.caption("Доля показаний датчиков (включая начальное наблюдение) в целевом диапазоне. "
                   "Электроэнергия оценена при постоянной мощности: кВт·ч = Вт × мин / 60 / 1000; "
                   "это не прогноз счёта и не доказательство экономии.")

    plot(sensor_chart(result, "temperature"))
    plot(sensor_chart(result, "humidity"))
    plot(sensor_chart(result, "illuminance"))
    st.markdown("**Состояния устройств**")
    plot(device_chart(result))

    st.markdown("**Журнал событий**")
    events = events_for_export(result)
    if events.empty:
        st.caption("Переключений не было.")
    else:
        st.dataframe(events, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(events)))

    st.markdown("**Скачать результаты**")
    c = st.columns(4)
    c[0].download_button("Показания и состояния (CSV)", to_csv_bytes(frame_for_export(result)),
                         "virtualiot_readings.csv", "text/csv", width="stretch")
    c[1].download_button("Журнал событий (CSV)", to_csv_bytes(events), "virtualiot_events.csv",
                         "text/csv", width="stretch")
    c[2].download_button("Конфигурация (JSON)", config_to_json(cfg), "virtualiot_config.json",
                         "application/json", width="stretch")
    c[3].download_button("Отчёт (HTML)", build_html_report(result, comparison).encode("utf-8"),
                         "virtualiot_report.html", "text/html", width="stretch")
    if comparison is not None:
        st.caption("HTML-отчёт включает сравнение сценариев, выполненное для тех же параметров.")


def render_room_tab(config: SimulationConfig | None, errors: list[str]) -> None:
    st.caption("Все показания смоделированы. Приложение не подключено к реальному оборудованию.")
    with st.container(border=True):
        render_preset_picker("room")
    render_inputs()
    if errors:
        st.error("Исправьте параметры:\n\n" + "\n".join(f"- {e}" for e in errors))
    if st.button("Запустить симуляцию", type="primary", disabled=config is None):
        assert config is not None
        try:
            st.session_state.result = (config, run_simulation(config))
        except ConfigError as exc:
            st.error("\n".join(exc.messages))

    stored = st.session_state.get("result")
    if stored is None:
        st.info("Задайте параметры и нажмите «Запустить симуляцию».")
        return
    stored_cfg, result = stored
    comp = st.session_state.get("comparison")
    comparison = comp[1] if comp is not None and comp[0] == stored_cfg.with_overrides(automation_enabled=True) else None
    render_results(result, stale=stored_cfg != config, comparison=comparison)


# ---------------------------------------------------------------- вкладка 2

def render_scenarios_tab(config: SimulationConfig | None) -> None:
    st.caption("Парное сравнение: одинаковые начальные условия, длительность, коэффициенты и "
               "последовательности шума датчиков. Данные смоделированы.")
    with st.container(border=True):
        render_preset_picker("scn")
        st.caption("Пресет загружает параметры на вкладку «Умное помещение»; сравнение использует текущие параметры.")
    if config is None:
        st.error("Параметры на вкладке «Умное помещение» содержат ошибки.")
        return
    st.markdown(
        f"Текущие условия: в помещении {num(config.initial_temperature)} °C и {num(config.initial_humidity, 0)} %, "
        f"на улице {num(config.outdoor_temperature)} °C и {num(config.outdoor_humidity, 0)} %, "
        f"естественный свет {num(config.natural_illuminance, 0)} лк, длительность {config.duration_min} мин, "
        f"зерно {config.seed}."
    )
    if st.button("Сравнить сценарии"):
        st.session_state.comparison = (config.with_overrides(automation_enabled=True), run_comparison(config))

    stored = st.session_state.get("comparison")
    if stored is not None:
        stored_cfg, comp = stored
        if stored_cfg != config.with_overrides(automation_enabled=True):
            st.warning("Параметры изменены после сравнения. Показаны прежние результаты — запустите сравнение заново.")
        st.markdown(f"**{BASELINE_LABEL}** (все устройства выключены) и **{AUTOMATED_LABEL}**")
        st.dataframe(comp.table, hide_index=True, width="stretch", height=36 * (len(comp.table) + 1) + 4)
        st.caption("Лучшие условия в помещении могут требовать дополнительной электроэнергии: автоматизация "
                   "здесь не является энергосбережением по определению. Энергия — оценка при постоянной мощности.")
        quantity = st.radio("Величина на графике", ["Температура", "Влажность", "Освещённость"], horizontal=True)
        plot(comparison_chart(comp, quantity))
        st.download_button("Скачать сравнение (CSV)", to_csv_bytes(comp.table), "virtualiot_comparison.csv", "text/csv")

    st.divider()
    st.markdown("**Эксперимент: узкая и широкая полоса гистерезиса**")
    st.caption("Оба прогона используют текущие параметры, автоматизацию и одинаковый шум; меняется только "
               "ширина полосы выбранного устройства вокруг общего центра.")
    devices = {"Обогреватель": "heater", "Вентиляция": "fan", "Лампа": "lamp"}
    defaults = {"heater": (21.0, 0.4, 3.0, "°C"), "fan": (55.0, 2.0, 16.0, "%"), "lamp": (300.0, 20.0, 150.0, "лк")}
    c = st.columns(4, vertical_alignment="bottom")
    device = devices[c[0].selectbox("Устройство", list(devices))]
    center0, narrow0, wide0, unit = defaults[device]
    center = c[1].number_input(f"Центр полосы, {unit}", value=center0, key=f"hx_c_{device}")
    narrow = c[2].number_input(f"Узкая полоса, {unit}", min_value=0.01, value=narrow0, key=f"hx_n_{device}")
    wide = c[3].number_input(f"Широкая полоса, {unit}", min_value=0.02, value=wide0, key=f"hx_w_{device}")
    if st.button("Провести эксперимент"):
        try:
            st.session_state.hyst = ((config, device, center, narrow, wide),
                                     run_hysteresis_experiment(config, device, center, narrow, wide))
        except ConfigError as exc:
            st.error("\n".join(exc.messages))
    stored = st.session_state.get("hyst")
    if stored is not None:
        key, exp = stored
        if key != (config, device, center, narrow, wide):
            st.warning("Параметры эксперимента изменены. Показаны прежние результаты.")
        st.dataframe(exp.table, hide_index=True, width="stretch")
        n_res, w_res = exp.narrow, exp.wide
        q = {"heater": "temperature", "fan": "humidity", "lamp": "illuminance"}[exp.device]
        st.markdown(
            f"Наблюдение: переключений {n_res.device_metrics[exp.device].switches} (узкая) и "
            f"{w_res.device_metrics[exp.device].switches} (широкая); доля времени в целевом диапазоне "
            f"{num(n_res.in_range_pct[q])} % и {num(w_res.in_range_pct[q])} %. "
            "Широкая полоса обычно уменьшает число переключений, но допускает большие отклонения величины; "
            "итог зависит от параметров и целевого диапазона, поэтому ни один вариант не является лучшим всегда."
        )
        plot(hysteresis_chart(exp))
        st.download_button("Скачать результаты эксперимента (CSV)", to_csv_bytes(exp.table),
                           "virtualiot_hysteresis.csv", "text/csv")


# ---------------------------------------------------------------- вкладка 3

def render_about_tab() -> None:
    st.markdown("""
#### Назначение
VirtualIoT — учебная виртуальная лаборатория. Она моделирует одно помещение с датчиками температуры,
влажности и освещённости и тремя устройствами: обогревателем, вентиляцией и лампой. Пользователь задаёт
правила автоматизации и наблюдает, как они влияют на смоделированную среду. **Все данные смоделированы,
реальное оборудование не подключается.**

**Целевая аудитория:** студенты, изучающие интернет вещей (IoT) и автоматизацию.

#### Архитектура
- **Виртуальные датчики** — измеряют состояние среды с добавлением шума (`simulation.py`).
- **Контроллер** — правила с гистерезисом (`controller.py`).
- **Устройства** — обогреватель, вентиляция, лампа; их состояние действует на интервале в 1 минуту.
- **Среда** — дискретная модель температуры, влажности и освещённости (`simulation.py`).

Порядок одного шага: 1) получить показания датчиков; 2) применить правила; 3) задать воздействие устройств
на следующий интервал; 4) продвинуть состояние среды; 5) записать результат.

#### Уравнения модели (шаг Δt = 1 мин)
""")
    st.latex(r"T_{t+1} = T_t + (k_T + k_{TV}\,v_t)\,(T_{out} - T_t) + h\,q_t")
    st.latex(r"H_{t+1} = H_t + (k_H + k_{HV}\,v_t)\,(H_{out} - H_t)")
    st.latex(r"E^{nat}_t = E_0\,\bigl(1 + a \sin(2\pi t / P)\bigr), \qquad E_t = E^{nat}_t + L\,l_{t-1}")
    st.markdown("""
Здесь \\(q, v, l \\in \\{0, 1\\}\\) — состояния обогревателя, вентиляции и лампы; \\(k\\) — коэффициенты
обмена с улицей, 1/мин; \\(h\\) — нагрев, °C/мин; \\(L\\) — вклад лампы, лк; \\(a\\) и \\(P\\) — амплитуда и
период колебаний естественного света. Для устойчивости проверяется \\(k_T + k_{TV} \\le 1\\) и
\\(k_H + k_{HV} \\le 1\\): так шаг релаксации не перескакивает через уличное значение.

Показание датчика = состояние + шум \\(\\sigma \\cdot z_t\\), где \\(z_t\\) — независимый воспроизводимый поток
для каждого датчика. Шум не меняет состояние среды. Измеренная влажность ограничивается 0–100 %,
освещённость — неотрицательными значениями.

#### Гистерезис
Устройство включается при выходе показания за один порог и выключается только после выхода за другой.
Между порогами состояние сохраняется, что предотвращает частые переключения из-за шума. Пример:
обогреватель включается ниже 20 °C и выключается выше 22 °C.

Лампа управляется по **отдельному датчику естественного света**, который не учитывает свет самой лампы.
Если бы контроллер смотрел на общую освещённость, включённая лампа сразу подняла бы показание выше верхнего
порога и выключилась, а затем снова включилась — возникли бы циклические переключения. Общая
освещённость помещения показывается отдельно и используется для оценки целевого диапазона.

#### Технологии
Python 3.11+, Streamlit, NumPy, pandas, Plotly, pytest. Работает локально, без облака и оборудования.

#### Практическая польза
- Безопасная проверка правил автоматизации до работы с реальными устройствами.
- Наглядная демонстрация гистерезиса, шума датчиков и компромисса «комфорт — электроэнергия».
- Воспроизводимые эксперименты: фиксированное зерно и экспорт результатов.

#### Ограничения
- Модель учебная и не предсказывает поведение реального здания.
- Влажность — приближение: не моделируются абсолютная влажность, конденсация и зависимость
  относительной влажности от температуры.
- Коэффициенты постоянны; не учитываются люди, солнечный нагрев, инерция стен и задержки устройств.
- Электроэнергия оценивается при постоянной мощности и не является прогнозом счёта.

#### Исследовательская гипотеза
*«Виртуальная модель позволяет исследовать влияние пороговых правил и гистерезиса на условия в помещении
и частоту переключения устройств.»*

**Воспроизводимый эксперимент** (вкладка «Сценарии», раздел «Эксперимент»):
1. Зафиксируйте начальные условия и зерно шума (например, пресет «Холодное помещение»).
2. Сравните узкую (0,4 °C) и широкую (3 °C) полосу гистерезиса с общим центром 21 °C.
3. Измерьте число переключений и долю времени в целевом диапазоне.
4. Скачайте фактические результаты в CSV.
5. Объясните компромисс: широкая полоса обычно снижает число переключений, но допускает большие
   колебания; результат зависит от параметров, и широкая полоса не всегда лучше.

#### Демонстрация за три минуты
1. На вкладке «Умное помещение» загрузите пресет «Холодное помещение».
2. Выключите автоматизацию (все устройства выключены) и запустите симуляцию — температура падает.
3. Включите автоматизацию и снова запустите симуляцию.
4. Покажите график температуры с порогами, график состояний и журнал переключений обогревателя.
5. На вкладке «Сценарии» нажмите «Сравнить сценарии» — время в диапазоне, переключения и электроэнергия.
6. Вернитесь на первую вкладку и скачайте HTML-отчёт.
""")
    st.caption(f"Версия модели {MODEL_VERSION} · шаг моделирования {TIME_STEP_MIN} мин")


# ---------------------------------------------------------------- main

def main() -> None:
    init_state()
    st.title("VirtualIoT")
    st.markdown(f"<p style='color:{MUTED};margin-top:-0.8rem'>Виртуальная лаборатория умного помещения</p>",
                unsafe_allow_html=True)

    config: SimulationConfig | None
    errors: list[str] = []
    try:
        config = config_from_widgets()
        config.validate()
    except ConfigError as exc:
        config, errors = None, exc.messages

    tab_room, tab_scn, tab_about = st.tabs(["Умное помещение", "Сценарии", "О проекте"])
    with tab_room:
        render_room_tab(config, errors)
    with tab_scn:
        render_scenarios_tab(config)
    with tab_about:
        render_about_tab()


main()
