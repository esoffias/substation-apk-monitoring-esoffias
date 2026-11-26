from __future__ import annotations

import html
import sqlite3
import textwrap
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.ensemble import IsolationForest


# =============================================================================
# ПОДСТАНЦИЯ АПК — ДЕМОНСТРАЦИОННАЯ ИИ-ПЛАТФОРМА МОНИТОРИНГА
# Запуск:
#   python -m pip install "streamlit>=1.40" pandas numpy plotly scikit-learn pillow
#   python -m streamlit run app.py
#
# Важно: приложение является учебной программной симуляцией. Оно не подключается
# к реальным датчикам, релейной защите, выключателям или исполнительным механизмам.
# Все измерения, аварии, события и записи оборудования в демонстрационной базе
# данных являются условными, пока пользователь не внесет собственные сведения.
# =============================================================================


# =============================================================================
# 1. ИКОНКА И НАСТРОЙКИ СТРАНИЦЫ
# =============================================================================


def create_favicon() -> Image.Image:
    """Создает favicon: буква S, переплетенная с неоновой молнией."""
    size = 128
    base = Image.new("RGBA", (size, size), (3, 5, 10, 255))
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(
        (5, 5, size - 6, size - 6),
        radius=28,
        fill=(8, 15, 29, 255),
        outline=(0, 240, 255, 190),
        width=3,
    )

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 91)
    except OSError:
        try:
            font = ImageFont.truetype("Arial.ttf", 91)
        except OSError:
            font = ImageFont.load_default()

    text = "S"
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size - text_width) / 2 - bbox[0] - 5
    text_y = (size - text_height) / 2 - bbox[1] - 8

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.text((text_x, text_y), text, font=font, fill=(0, 240, 255, 230))
    bolt = [(83, 12), (59, 57), (77, 57), (43, 116), (55, 73), (39, 73)]
    glow_draw.line(bolt, fill=(0, 240, 255, 255), width=10, joint="curve")
    blurred = glow.filter(ImageFilter.GaussianBlur(9))
    base = Image.alpha_composite(base, blurred)

    foreground = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    foreground_draw = ImageDraw.Draw(foreground)
    foreground_draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(226, 252, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 240, 255, 255),
    )
    foreground_draw.line(bolt, fill=(0, 240, 255, 255), width=6, joint="curve")
    foreground_draw.line(bolt, fill=(225, 255, 255, 255), width=2, joint="curve")
    return Image.alpha_composite(base, foreground)


st.set_page_config(
    page_title="Подстанция АПК | Интеллектуальный мониторинг",
    page_icon=create_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Демонстрационная платформа мониторинга электрических подстанций агропромышленного комплекса.",
    },
)


# =============================================================================
# 2. КОНСТАНТЫ И ЦВЕТОВАЯ СИСТЕМА
# =============================================================================

APP_TITLE = "МОНИТОРИНГ ПОДСТАНЦИЙ АПК"
AUTHOR_FULL_NAME = "Софья Александровна Болтикова"
FACILITIES = ["Ферма «Рассвет»", "Элеватор «Заря»", "Птицефабрика «Юг»"]
DB_PATH = Path(__file__).resolve().with_name("substation_demo.db")

COLORS = {
    "background": "#03050A",
    "panel": "#0A0F1D",
    "panel_alt": "#111827",
    "white": "#F8FAFC",
    "indigo": "#4F46E5",
    "orange": "#FF8C00",
    "cyan": "#00F0FF",
    "yellow": "#F59E0B",
    "green": "#22C55E",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
}

# Демонстрационные границы интерфейса. Это не уставки реальной защиты.
CURRENT_WARNING_A = 20.0
CURRENT_CRITICAL_A = 40.0
VOLTAGE_LOW_V = 370.0
VOLTAGE_HIGH_V = 390.0
TEMPERATURE_NORMAL_HIGH_C = 70.0
TEMPERATURE_WARNING_C = 70.0
TEMPERATURE_CRITICAL_C = 90.0

PARAMETERS: dict[str, dict[str, Any]] = {
    "Ток": {
        "column": "Ток",
        "db_column": "current_a",
        "symbol": "I",
        "unit": "А",
        "color": "#00F0FF",
        "range": [0, 65],
        "normal": (10, 20),
        "warning": (20, 40),
        "critical": (40, 65),
        "description": "Ток нагрузки",
    },
    "Напряжение": {
        "column": "Напряжение",
        "db_column": "voltage_v",
        "symbol": "U",
        "unit": "В",
        "color": "#9B8CFF",
        "range": [340, 420],
        "normal": (370, 390),
        "warning_low": (355, 370),
        "warning_high": (390, 405),
        "critical_low": (340, 355),
        "critical_high": (405, 420),
        "description": "Напряжение сети",
    },
    "Температура": {
        "column": "Температура",
        "db_column": "temperature_c",
        "symbol": "T",
        "unit": "°C",
        "color": "#FFC266",
        "range": [30, 130],
        "normal": (50, 70),
        "warning": (70, 90),
        "critical": (90, 130),
        "description": "Температура оборудования",
    },
}


# =============================================================================
# 3. CSS — ТЕМНЫЙ ИНТЕРФЕЙС, НЕОНОВЫЕ РАМКИ И СВЕТЛЫЕ КАРТОЧКИ
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Montserrat:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #03050A;
        --panel: #0A0F1D;
        --panel-alt: #111827;
        --indigo: #4F46E5;
        --orange: #FF8C00;
        --cyan: #00F0FF;
        --text: #E5E7EB;
        --muted: #94A3B8;
        --primary-color: #FF8C00;
    }
    html { accent-color: #FF8C00; }

    /* Единая типографика окон и универсальная неоновая прокрутка. */
    * {
        scrollbar-width: thin;
        scrollbar-color: #7667F4 #121936;
    }
    *::-webkit-scrollbar { width: 13px; height: 13px; }
    *::-webkit-scrollbar-track {
        background: linear-gradient(180deg, #111532 0%, #172448 52%, #10172E 100%);
        border: 1px solid rgba(0, 240, 255, 0.22);
        border-radius: 999px;
    }
    *::-webkit-scrollbar-thumb {
        min-height: 34px;
        border: 2px solid #11172D;
        border-radius: 999px;
        background-color: #4F46E5;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='20' viewBox='0 0 12 20'%3E%3Cpath d='M7.2 0.8L1.4 10.8h3.8l-0.9 8.4 7-11.5H7.2z' fill='%23FFC266' stroke='%23FF8C00' stroke-width='0.8' stroke-linejoin='round'/%3E%3C/svg%3E"),
            linear-gradient(180deg, #4F46E5 0%, #8B5CF6 44%, #00CFEF 100%);
        background-repeat: no-repeat, no-repeat;
        background-position: center, center;
        background-size: 11px 19px, 100% 100%;
        box-shadow: 0 0 8px rgba(0, 240, 255, 0.42), 0 0 12px rgba(139, 92, 246, 0.30);
    }
    *::-webkit-scrollbar-thumb:hover {
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='20' viewBox='0 0 12 20'%3E%3Cpath d='M7.2 0.8L1.4 10.8h3.8l-0.9 8.4 7-11.5H7.2z' fill='%23FFE08A' stroke='%23FF8C00' stroke-width='0.8' stroke-linejoin='round'/%3E%3C/svg%3E"),
            linear-gradient(180deg, #6257F4 0%, #A46AF7 44%, #00F0FF 100%);
        box-shadow: 0 0 11px rgba(0, 240, 255, 0.70), 0 0 14px rgba(255, 140, 0, 0.25);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at center, #0A0F1D 0%, #03050A 68%, #000000 100%) !important;
        color: var(--text) !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    .stApp {
        box-shadow: inset 0 0 190px 42px rgba(0, 0, 0, 0.82);
    }

    [data-testid="stHeader"], #MainMenu, header, footer {
        visibility: hidden;
        height: 0;
    }

    .block-container {
        max-width: 1720px;
        padding-top: 1.0rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label, span, div {
        font-family: 'Montserrat', sans-serif;
    }

    .hero-title {
        color: #F1F5F9;
        text-align: center;
        font-weight: 800;
        letter-spacing: 1.15px;
        font-size: clamp(1.45rem, 2.7vw, 2.55rem);
        line-height: 1.2;
        text-shadow: 0 0 10px rgba(79, 70, 229, 0.85),
                     0 0 27px rgba(79, 70, 229, 0.34);
        margin: 0.1rem 0 0.25rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.17em;
        font-size: 0.69rem;
        margin: 0 0 1.1rem;
    }

    .section-heading {
        color: #C7D2FE;
        font-size: 1.04rem;
        font-weight: 800;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        border-left: 3px solid var(--orange);
        padding-left: 0.75rem;
        margin: 0.55rem 0 0.95rem;
    }

    .small-muted {
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.55;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(16, 26, 44, 0.96), rgba(7, 12, 23, 0.98));
        border: 1px solid rgba(79, 70, 229, 0.62);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.35rem 0 0.75rem;
        color: #E5E7EB;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.10);
    }

    .glass-card h4 {
        color: #A5B4FC;
        margin: 0 0 0.55rem;
        font-size: 0.94rem;
        font-weight: 800;
    }

    .glass-card p, .glass-card li {
        color: #CBD5E1;
        font-size: 0.81rem;
        line-height: 1.6;
        margin-bottom: 0.45rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        color: #DFFBFF;
        background: rgba(0, 240, 255, 0.06);
        font-size: 0.73rem;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        display: inline-block;
        border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 9px currentColor;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .neon-number {
        color: #00F0FF;
        font-weight: 800;
        text-shadow: 0 0 8px rgba(0, 240, 255, 0.35);
    }

    [data-testid="stMetric"] {
        background: rgba(10, 15, 29, 0.78);
        border: 1px solid rgba(79, 70, 229, 0.48);
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 0 12px rgba(79, 70, 229, 0.08);
        min-height: 94px;
    }

    [data-testid="stMetricLabel"] { color: #A5B4FC !important; }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; }

    /* Неоновая рамка графиков и лёгкое увеличение при наведении. */
    [data-testid="stPlotlyChart"] {
        position: relative;
        z-index: 1;
        border: 1.5px solid transparent !important;
        border-radius: 12px !important;
        background: linear-gradient(rgba(10,15,29,0.94), rgba(10,15,29,0.94)) padding-box,
                    linear-gradient(135deg, #FF8C00 0%, #FFC266 42%, #FF8C00 100%) border-box !important;
        box-shadow: 0 0 9px rgba(255, 140, 0, 0.55),
                    0 0 22px rgba(255, 140, 0, 0.20),
                    inset 0 0 12px rgba(255, 140, 0, 0.06);
        padding: 0.28rem;
        transition: transform 0.20s ease, box-shadow 0.20s ease;
        cursor: pointer;
        transform-origin: center center;
    }
    [data-testid="stPlotlyChart"]:hover {
        z-index: 5;
        transform: scale(1.018);
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.78),
                    0 0 31px rgba(255, 140, 0, 0.30),
                    inset 0 0 15px rgba(255, 140, 0, 0.10);
    }

    /* Навигация: активный раздел подсвечивается фирменным оранжевым. */
    [data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        align-items: center;
    }
    [data-testid="stRadio"] [role="radio"] {
        border: 1px solid rgba(79, 70, 229, 0.58) !important;
        border-radius: 10px !important;
        padding: 0.52rem 0.90rem !important;
        background: rgba(10, 15, 29, 0.90) !important;
        color: #C7D2FE !important;
        transition: all 0.18s ease;
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        border-color: #FF8C00 !important;
        background: rgba(255, 140, 0, 0.13) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 11px rgba(255, 140, 0, 0.40), inset 0 0 9px rgba(255, 140, 0, 0.08);
    }
    [data-testid="stRadio"] [role="radio"]:hover {
        border-color: #FF8C00 !important;
        box-shadow: 0 0 10px rgba(255, 140, 0, 0.25);
    }
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"]:hover {
        background: rgba(255, 140, 0, 0.16) !important;
        color: #FFFFFF !important;
    }
    [data-baseweb="input"]:focus-within,
    [data-baseweb="select"]:focus-within,
    [data-testid="stDateInput"]:focus-within {
        border-color: #FF8C00 !important;
        box-shadow: 0 0 0 1px rgba(255,140,0,0.30), 0 0 10px rgba(255,140,0,0.18) !important;
    }
    [data-baseweb="tab-highlight"] { background-color: #FF8C00 !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom-color: rgba(255,140,0,0.45); }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background: rgba(255, 140, 0, 0.11) !important;
        border-bottom: 2px solid #FF8C00 !important;
        box-shadow: 0 0 10px rgba(255, 140, 0, 0.16);
    }

    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border: 1px solid rgba(0, 240, 255, 0.52);
        border-radius: 9px;
        background: linear-gradient(120deg, rgba(79, 70, 229, 0.24), rgba(255, 140, 0, 0.10));
        color: #F8FAFC;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        min-height: 2.55rem;
        transition: all 0.18s ease;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        border-color: #FF8C00 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 13px rgba(255, 140, 0, 0.34);
        transform: translateY(-1px);
    }
    [data-testid="stButton"] button:focus,
    [data-testid="stButton"] button:active,
    [data-testid="stDownloadButton"] button:focus,
    [data-testid="stFormSubmitButton"] button:focus {
        border-color: #FF8C00 !important;
        box-shadow: 0 0 0 1px rgba(255,140,0,0.35), 0 0 13px rgba(255,140,0,0.30) !important;
        outline: none !important;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stDateInput"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stNumberInput"] label {
        color: #C7D2FE !important;
        font-weight: 600;
        font-size: 0.82rem;
    }

    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #101A2C !important;
        color: #F8FAFC !important;
        border-color: #4F46E5 !important;
        border-radius: 8px !important;
    }

    /* Ползунки параметров: индиго → фиолетовый → неоново-голубой. */
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #4F46E5 0%, #8B5CF6 48%, #00F0FF 100%) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: linear-gradient(145deg, #FFC266, #FF8C00) !important;
        border: 2px solid #FFD28A !important;
        box-shadow: 0 0 8px rgba(255, 140, 0, 0.80), 0 0 14px rgba(255, 194, 102, 0.38) !important;
    }
    [data-testid="stSlider"] [role="slider"]::after {
        content: 'ϟ';
        color: #4B2B08;
        font-size: 15px;
        font-weight: 900;
        line-height: 1;
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        text-shadow: 0 0 3px rgba(255,245,210,0.8);
        pointer-events: none;
    }

    /* Модальные окна: компактные, центральные, с затемнением фона. */
    [data-testid="stDialog"] {
        background: rgba(0, 2, 10, 0.82) !important;
        backdrop-filter: blur(5px);
    }
    [data-testid="stDialog"] > div > div,
    [data-testid="stDialog"] [role="dialog"] {
        background: linear-gradient(145deg, #0D1530, #090E1D) !important;
        border: 1.5px solid transparent !important;
        border-radius: 17px !important;
        background-image: linear-gradient(145deg, #0D1530, #090E1D),
                          linear-gradient(135deg, #00F0FF 0%, #6657F5 48%, #A855F7 100%) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        box-shadow: 0 0 22px rgba(0, 240, 255, 0.20),
                    0 0 30px rgba(124, 58, 237, 0.24) !important;
        max-height: 82vh !important;
        overflow-y: auto !important;
    }
    [data-testid="stDialog"] h1,
    [data-testid="stDialog"] h2,
    [data-testid="stDialog"] h3,
    [data-testid="stDialog"] h4 { color: #DCE6FF !important; }
    [data-testid="stDialog"] p,
    [data-testid="stDialog"] label { color: #D4DDF0 !important; }
    [data-testid="stDialog"] [data-testid="stButton"] button {
        color: #F8FAFC !important;
        border-color: #7166E8 !important;
        background: #121A36 !important;
    }
    [data-testid="stDialog"]:has(.author-dialog-panel) [role="dialog"] {
        width: min(700px, 92vw) !important;
        max-width: 700px !important;
    }
    [data-testid="stDialog"]:has(.author-dialog-panel) h1,
    [data-testid="stDialog"]:has(.author-dialog-panel) h2 {
        color: #172554 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Компактные справочные секции вместо растянутых светлых карточек. */
    .reference-section {
        background: linear-gradient(145deg, rgba(35, 30, 91, 0.96), rgba(10, 15, 29, 0.98));
        border: 1px solid rgba(0, 240, 255, 0.30);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.35rem 0 0.7rem;
        box-shadow: 0 0 12px rgba(79, 70, 229, 0.14);
    }
    .reference-section h4 { color: #C8C1FF !important; margin: 0 0 0.45rem; font-weight: 800; }
    .reference-section p, .reference-section li { color: #E2E8F0 !important; font-size: 0.80rem; line-height: 1.55; margin-bottom: 0.35rem; }

    /* Таблицы требований: индиго-фиолетовое поле и неоновый код требования. */
    .docs-table-wrap {
        width: 100%;
        max-height: 64vh;
        overflow: auto;
        border: 1.5px solid transparent;
        border-radius: 12px;
        background: linear-gradient(#17143B, #17143B) padding-box,
                    linear-gradient(135deg, #00F0FF, #5B4BFF, #A855F7, #00F0FF) border-box;
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.24), 0 0 18px rgba(124, 58, 237, 0.22);
        padding: 0.30rem;
    }
    table.requirements-table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        min-width: 680px;
        background: #17143B;
        color: #F8FAFC;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.79rem;
    }
    table.requirements-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: linear-gradient(110deg, #35217F, #4F46E5 60%, #5737A9);
        color: #FFFFFF;
        padding: 0.82rem 0.72rem;
        text-align: left;
        border: 1px solid rgba(174, 160, 255, 0.48);
        text-shadow: 0 0 7px rgba(0, 240, 255, 0.18);
    }
    table.requirements-table td {
        border: 1px solid rgba(112, 104, 235, 0.52);
        color: #F1F5F9;
        background: #17143B;
        padding: 0.76rem 0.70rem;
        vertical-align: top;
        line-height: 1.55;
    }
    table.requirements-table tr:nth-child(even) td { background: #211B55; }
    table.requirements-table td:first-child {
        color: #D9B8FF;
        background: rgba(102, 67, 190, 0.35) !important;
        font-weight: 800;
        text-shadow: 0 0 8px rgba(168, 85, 247, 0.68);
        border-right-color: #A855F7;
        white-space: nowrap;
    }
    table.requirements-table td strong { color: #FFC266; }
    table.requirements-table td:first-child strong {
        color: #D9B8FF !important;
        text-shadow: 0 0 8px rgba(168, 85, 247, 0.78);
    }

    .role-restrictions { list-style: none; padding-left: 0.2rem; color: #E2E8F0; }
    .role-restrictions li {
        position: relative;
        padding-left: 1.45rem;
        margin: 0.55rem 0;
        color: #E2E8F0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .role-restrictions li::before {
        content: '✕'; color: #FFC266; font-weight: 800; position: absolute; left: 0;
    }

    /* Три светлые карточки в документации. */
    .st-key-docs-requirements-card,
    .st-key-docs-roles-card,
    .st-key-docs-glossary-card {
        background: #F8FAFC;
        border: 1.5px solid #4F46E5;
        border-radius: 16px;
        padding: 1rem 1rem 0.75rem;
        min-height: 290px;
        box-shadow: 0 0 13px rgba(79, 70, 229, 0.14);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .st-key-docs-requirements-card:hover,
    .st-key-docs-roles-card:hover,
    .st-key-docs-glossary-card:hover {
        border-color: #00F0FF;
        box-shadow: 0 0 17px rgba(0, 240, 255, 0.28);
    }
    .doc-card-title {
        color: #1E293B;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.35;
        margin: 0.35rem 0 0.45rem;
    }
    .doc-card-desc {
        color: #64748B;
        font-size: 0.76rem;
        line-height: 1.5;
        min-height: 48px;
        margin-bottom: 0.8rem;
    }
    .doc-card-symbol {
        color: #4F46E5;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1;
    }
    .st-key-docs-requirements-card [data-testid="stButton"] button,
    .st-key-docs-roles-card [data-testid="stButton"] button,
    .st-key-docs-glossary-card [data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #4F46E5 !important;
        border-radius: 10px !important;
        min-height: 56px !important;
        white-space: normal !important;
        box-shadow: 0 3px 10px rgba(79, 70, 229, 0.08);
        font-size: 0.78rem !important;
    }
    .st-key-docs-requirements-card [data-testid="stButton"] button:hover,
    .st-key-docs-roles-card [data-testid="stButton"] button:hover,
    .st-key-docs-glossary-card [data-testid="stButton"] button:hover {
        background: #EEF2FF !important;
        border-color: #00F0FF !important;
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.32) !important;
    }

    .parameter-chip {
        display: inline-flex;
        width: 2.25rem;
        height: 2.25rem;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 1.5px solid #00F0FF;
        color: #DFFBFF;
        background: rgba(0, 240, 255, 0.07);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
        font-weight: 800;
        margin-right: 0.6rem;
    }

    .author-dialog-panel {
        font-family: 'DM Sans', sans-serif !important;
        color: #172554;
        background: linear-gradient(145deg, #FBFDFF, #EEF4FF);
        border: 2px solid transparent;
        border-radius: 15px;
        background-image: linear-gradient(145deg, #FBFDFF, #EEF4FF),
                          linear-gradient(125deg, #00F0FF 0%, #4F46E5 48%, #A855F7 100%);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        padding: 1.15rem 1.25rem;
        margin: 0.1rem auto 0.45rem;
        max-width: 650px;
        box-shadow: 0 0 18px rgba(0, 240, 255, 0.23), 0 0 24px rgba(124, 58, 237, 0.18);
    }
    .author-dialog-panel .author-kicker {
        color: #4F46E5;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .author-dialog-panel h2 {
        color: #172554 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.25rem;
        line-height: 1.25;
        margin: 0 0 0.55rem;
    }
    .author-dialog-panel h3 {
        color: #243B75 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.91rem;
        margin: 0.8rem 0 0.25rem;
    }
    .author-dialog-panel p {
        color: #24324F !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.79rem;
        line-height: 1.53;
        margin: 0.25rem 0 0.48rem;
    }
    .author-dialog-panel .author-techline {
        color: #334155;
        font-size: 0.75rem;
        line-height: 1.55;
        background: rgba(79, 70, 229, 0.055);
        border: 1px solid rgba(79, 70, 229, 0.14);
        border-radius: 9px;
        padding: 0.55rem 0.7rem;
        margin-top: 0.4rem;
    }
    .author-dialog-panel .author-footer {
        border-top: 1px solid rgba(79, 70, 229, 0.24);
        margin-top: 0.8rem;
        padding-top: 0.7rem;
        color: #172554;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.79rem;
    }
    .author-dialog-panel .author-footer strong {
        color: #172554;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
    }
    .author-dialog-panel .author-footer small { color: #64748B; }

    .emergency-alert {
        border: 1px solid rgba(255, 65, 95, 0.96);
        border-radius: 13px;
        padding: 0.9rem 1.05rem;
        color: #FFE8ED;
        background: linear-gradient(120deg, rgba(76, 10, 27, 0.92), rgba(25, 9, 23, 0.96));
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.42), inset 0 0 16px rgba(239, 68, 68, 0.09);
        animation: emergencyPulse 2.1s ease-in-out infinite;
        transform-origin: center;
    }
    .emergency-alert strong { color: #FFFFFF; letter-spacing: 0.035em; }
    .emergency-alert p { color: #FFD8E0; margin: 0.35rem 0 0; font-size: 0.82rem; line-height: 1.5; }
    @keyframes emergencyPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 10px rgba(239, 68, 68, 0.40), inset 0 0 12px rgba(239, 68, 68, 0.08); }
        50% { transform: scale(1.012); box-shadow: 0 0 20px rgba(239, 68, 68, 0.72), 0 0 30px rgba(239, 68, 68, 0.24), inset 0 0 17px rgba(239, 68, 68, 0.13); }
    }
    .st-key-archive-jump [data-testid="stButton"] button {
        border: 1.5px solid #00F0FF !important;
        background: linear-gradient(110deg, rgba(79,70,229,0.24), rgba(0,240,255,0.11)) !important;
        box-shadow: 0 0 12px rgba(0,240,255,0.28);
    }
    .st-key-archive-jump [data-testid="stButton"] button:hover {
        transform: scale(1.025);
        box-shadow: 0 0 18px rgba(0,240,255,0.65) !important;
    }

    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid rgba(79, 70, 229, 0.38); border-radius: 10px; overflow: hidden; }
    hr { border-color: rgba(79, 70, 229, 0.34); }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .st-key-docs-requirements-card,
        .st-key-docs-roles-card,
        .st-key-docs-glossary-card { min-height: auto; }
    }


    /* ================================================================
       ЕДИНАЯ ФИНАЛЬНАЯ ТЕМА: НЕОНОВЫЙ СКРОЛЛБАР, ОРАНЖЕВАЯ НАВИГАЦИЯ,
       ЦЕНТРАЛЬНЫЕ ДИАЛОГИ, ТАБЛИЦЫ И ИНТЕРАКТИВНЫЕ ГРАФИКИ.
       ================================================================ */
    :root, html, body {
        accent-color: #FF8C00 !important;
        --primary-color: #FF8C00 !important;
        --st-primary-color: #FF8C00 !important;
    }
    input[type="radio"], input[type="checkbox"] {
        accent-color: #FF8C00 !important;
    }

    /* Вертикальная и горизонтальная прокрутка во всех основных областях. */
    * {
        scrollbar-width: thin !important;
        scrollbar-color: #8271F5 #11172D !important;
    }
    *::-webkit-scrollbar,
    [data-testid="stDialog"]::-webkit-scrollbar,
    [data-testid="stDialog"] *::-webkit-scrollbar,
    .docs-table-wrap::-webkit-scrollbar,
    .reference-section::-webkit-scrollbar {
        width: 14px !important;
        height: 14px !important;
    }
    *::-webkit-scrollbar-track,
    [data-testid="stDialog"] *::-webkit-scrollbar-track,
    .docs-table-wrap::-webkit-scrollbar-track,
    .reference-section::-webkit-scrollbar-track {
        background: linear-gradient(180deg, #111532 0%, #22205A 48%, #101A36 100%) !important;
        border: 1px solid rgba(0, 240, 255, 0.42) !important;
        border-radius: 999px !important;
    }
    *::-webkit-scrollbar-thumb,
    [data-testid="stDialog"] *::-webkit-scrollbar-thumb,
    .docs-table-wrap::-webkit-scrollbar-thumb,
    .reference-section::-webkit-scrollbar-thumb {
        min-height: 36px !important;
        border: 2px solid #11172D !important;
        border-radius: 999px !important;
        background-color: #4F46E5 !important;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='22' viewBox='0 0 14 22'%3E%3Cpath d='M8.4 0.8L1.5 12h4.7l-1 9.1 7.3-12H7.8z' fill='%23FFD166' stroke='%23FF8C00' stroke-width='1.1' stroke-linejoin='round'/%3E%3C/svg%3E"),
            linear-gradient(180deg, #4F46E5 0%, #8B5CF6 48%, #00E5FF 100%) !important;
        background-repeat: no-repeat, no-repeat !important;
        background-position: center center, center center !important;
        background-size: 12px 20px, 100% 100% !important;
        box-shadow: 0 0 9px rgba(0,240,255,.52), 0 0 15px rgba(139,92,246,.46) !important;
    }
    *::-webkit-scrollbar-thumb:hover,
    [data-testid="stDialog"] *::-webkit-scrollbar-thumb:hover,
    .docs-table-wrap::-webkit-scrollbar-thumb:hover,
    .reference-section::-webkit-scrollbar-thumb:hover {
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='22' viewBox='0 0 14 22'%3E%3Cpath d='M8.4 0.8L1.5 12h4.7l-1 9.1 7.3-12H7.8z' fill='%23FFE29A' stroke='%23FF9A1F' stroke-width='1.2' stroke-linejoin='round'/%3E%3C/svg%3E"),
            linear-gradient(180deg, #665CF7 0%, #A56BFA 48%, #00F0FF 100%) !important;
        box-shadow: 0 0 12px rgba(0,240,255,.78), 0 0 18px rgba(139,92,246,.62) !important;
    }

    /* Красный стандартный акцент Streamlit заменён фирменным оранжевым.
       Красный оставлен только для отдельного блока аварийного сценария. */
    [data-testid="stRadio"] [role="radio"],
    [data-testid="stCheckbox"] label,
    [data-testid="stTabs"] button[role="tab"],
    [data-baseweb="tab"] {
        --primary-color: #FF8C00 !important;
        --st-primary-color: #FF8C00 !important;
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        border: 1.5px solid #FF8C00 !important;
        background: linear-gradient(110deg, rgba(255,140,0,.17), rgba(255,194,102,.07)) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 9px rgba(255,140,0,.42), 0 0 18px rgba(255,140,0,.22) !important;
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child,
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] svg,
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] input {
        border-color: #FF8C00 !important;
        color: #FF8C00 !important;
        accent-color: #FF8C00 !important;
        fill: #FF8C00 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-baseweb="tab-highlight"] {
        background: #FF8C00 !important;
        box-shadow: 0 0 10px rgba(255,140,0,.65) !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FF8C00 !important;
        box-shadow: 0 0 12px rgba(255,140,0,.24) !important;
    }
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"]:hover {
        background: rgba(255,140,0,.20) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stButton"] button:focus-visible,
    [data-testid="stDownloadButton"] button:focus-visible,
    [data-testid="stFormSubmitButton"] button:focus-visible,
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    [data-testid="stDateInput"]:focus-within,
    [data-testid="stTextInput"]:focus-within,
    [data-testid="stNumberInput"]:focus-within {
        outline: 1px solid #FF8C00 !important;
        border-color: #FF8C00 !important;
        box-shadow: 0 0 0 1px rgba(255,140,0,.32), 0 0 13px rgba(255,140,0,.26) !important;
    }

    /* Все диалоги: затемнённая подложка, окно в центре, не на весь экран. */
    div[data-testid="stDialog"] {
        position: fixed !important;
        inset: 0 !important;
        z-index: 99999 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100vw !important;
        height: 100vh !important;
        max-width: 100vw !important;
        max-height: 100vh !important;
        padding: 18px !important;
        background: rgba(0, 2, 12, 0.86) !important;
        backdrop-filter: blur(7px) brightness(.48) saturate(.72) !important;
        -webkit-backdrop-filter: blur(7px) brightness(.48) saturate(.72) !important;
    }
    div[data-testid="stDialog"] [role="dialog"] {
        box-sizing: border-box !important;
        margin: auto !important;
        max-height: 84vh !important;
        overflow-y: auto !important;
        border: 2px solid transparent !important;
        border-radius: 18px !important;
        background:
            linear-gradient(145deg, #0B1230, #080D1D) padding-box,
            linear-gradient(125deg, #00F0FF 0%, #4F46E5 42%, #A855F7 72%, #00F0FF 100%) border-box !important;
        box-shadow: 0 0 18px rgba(0,240,255,.32), 0 0 34px rgba(124,58,237,.25), 0 22px 70px rgba(0,0,0,.58) !important;
    }
    div[data-testid="stDialog"] [role="dialog"] h1,
    div[data-testid="stDialog"] [role="dialog"] h2,
    div[data-testid="stDialog"] [role="dialog"] h3,
    div[data-testid="stDialog"] [role="dialog"] h4 {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Авторское окно — светлая компактная карточка с тёмно-синим текстом. */
    div[data-testid="stDialog"]:has(.author-dialog-panel) [role="dialog"] {
        width: min(540px, calc(100vw - 36px)) !important;
        max-width: 540px !important;
        max-height: 82vh !important;
        padding: 1.25rem 1.35rem !important;
        background:
            linear-gradient(145deg, #FBFDFF 0%, #EEF4FF 100%) padding-box,
            linear-gradient(125deg, #00F0FF 0%, #4F46E5 44%, #A855F7 76%, #00F0FF 100%) border-box !important;
        box-shadow: 0 0 18px rgba(0,240,255,.42), 0 0 30px rgba(124,58,237,.32), 0 22px 70px rgba(0,0,0,.62) !important;
    }
    div[data-testid="stDialog"]:has(.author-dialog-panel) [role="dialog"] h1,
    div[data-testid="stDialog"]:has(.author-dialog-panel) [role="dialog"] h2,
    div[data-testid="stDialog"]:has(.author-dialog-panel) [role="dialog"] h3 {
        color: #14285B !important;
    }
    .author-dialog-panel {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #172B5B !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .author-dialog-panel .author-kicker {
        color: #6554C8 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: .66rem !important;
        font-weight: 800 !important;
        letter-spacing: .15em !important;
        margin: 0 0 .42rem !important;
    }
    .author-dialog-panel h2 {
        color: #14285B !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.28rem !important;
        line-height: 1.22 !important;
        margin: 0 0 .55rem !important;
    }
    .author-dialog-panel h3 {
        color: #243B75 !important;
        font-size: .91rem !important;
        margin: .72rem 0 .18rem !important;
    }
    .author-dialog-panel p {
        color: #24324F !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: .79rem !important;
        line-height: 1.48 !important;
        margin: .20rem 0 .42rem !important;
    }
    .author-dialog-panel .author-intro {
        color: #172B5B !important;
        font-weight: 600 !important;
    }
    .author-dialog-panel .author-techline {
        color: #24324F !important;
        font-size: .75rem !important;
        line-height: 1.5 !important;
        background: rgba(79,70,229,.055) !important;
        border: 1px solid rgba(79,70,229,.20) !important;
        border-radius: 9px !important;
        padding: .56rem .68rem !important;
        margin-top: .38rem !important;
    }
    .author-dialog-panel .author-note {
        font-size: .73rem !important;
        color: #4B5875 !important;
    }
    .author-dialog-panel .author-footer {
        display: flex !important;
        flex-direction: column !important;
        gap: .15rem !important;
        border-top: 1px solid rgba(79,70,229,.25) !important;
        margin-top: .72rem !important;
        padding-top: .62rem !important;
        color: #172B5B !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: .68rem !important;
        letter-spacing: .08em !important;
    }
    .author-dialog-panel .author-footer strong {
        color: #14285B !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: .98rem !important;
        letter-spacing: 0 !important;
    }
    .author-dialog-panel .author-footer small {
        color: #64748B !important;
        font-size: .70rem !important;
        letter-spacing: 0 !important;
    }
    .author-dialog-panel .author-footer a {
        color: #4F46E5 !important;
        text-decoration: none !important;
        font-weight: 700 !important;
    }
    .author-dialog-panel .author-footer a:hover {
        color: #0891B2 !important;
        text-decoration: underline !important;
    }

    /* Документация и правила используют единый тёмный индиго-фиолетовый стиль. */
    .reference-section {
        background: linear-gradient(145deg, rgba(32,25,91,.98), rgba(10,15,35,.99)) !important;
        border: 1.5px solid transparent !important;
        background-image:
            linear-gradient(145deg, rgba(32,25,91,.98), rgba(10,15,35,.99)),
            linear-gradient(125deg, rgba(0,240,255,.78), rgba(79,70,229,.90), rgba(168,85,247,.90)) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        border-radius: 12px !important;
        box-shadow: 0 0 12px rgba(0,240,255,.16), 0 0 18px rgba(124,58,237,.18) !important;
    }
    .reference-section h4 { color: #D8D2FF !important; }
    .reference-section p, .reference-section li { color: #F1F5FF !important; }

    /* Таблицы требований: индиго фон, белый текст, яркая фиолетовая колонка ID. */
    .docs-table-wrap {
        max-height: 64vh !important;
        overflow: auto !important;
        border: 2px solid transparent !important;
        border-radius: 13px !important;
        background: linear-gradient(#17143B,#17143B) padding-box,
                    linear-gradient(135deg,#00F0FF 0%,#5750F5 38%,#A855F7 72%,#00F0FF 100%) border-box !important;
        box-shadow: 0 0 15px rgba(0,240,255,.32), 0 0 24px rgba(124,58,237,.27) !important;
        padding: .28rem !important;
    }
    table.requirements-table {
        background: #17143B !important;
        color: #FFFFFF !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
    }
    table.requirements-table th {
        background: linear-gradient(110deg,#38207E 0%,#4F46E5 55%,#673AB7 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(0,240,255,.52) !important;
        text-shadow: 0 0 8px rgba(0,240,255,.34) !important;
    }
    table.requirements-table td {
        color: #F8FAFC !important;
        background: #17143B !important;
        border: 1px solid rgba(105,92,235,.68) !important;
    }
    table.requirements-table tr:nth-child(even) td {
        background: #211B55 !important;
    }
    table.requirements-table td:first-child {
        color: #FFFFFF !important;
        background: linear-gradient(135deg,#5722A6,#7535C8) !important;
        border-right: 1px solid #C084FC !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        text-shadow: 0 0 7px rgba(216,180,254,.84), 0 0 13px rgba(168,85,247,.78) !important;
        box-shadow: inset 0 0 12px rgba(192,132,252,.20) !important;
        white-space: nowrap !important;
    }
    table.requirements-table td strong { color: #FFFFFF !important; }

    /* Графики: именно неоновая рамка, а не плоская оранжевая линия. */
    [data-testid="stPlotlyChart"] {
        border: 2px solid transparent !important;
        border-radius: 13px !important;
        background: linear-gradient(rgba(7,12,24,.97),rgba(7,12,24,.97)) padding-box,
                    linear-gradient(135deg,#FFB84D 0%,#FF8C00 28%,#FFD166 52%,#FF8C00 78%,#FFB84D 100%) border-box !important;
        box-shadow: 0 0 8px rgba(255,140,0,.72),
                    0 0 18px rgba(255,140,0,.48),
                    0 0 30px rgba(255,140,0,.20),
                    inset 0 0 13px rgba(255,140,0,.10) !important;
        transition: transform .22s ease, box-shadow .22s ease, filter .22s ease !important;
        transform-origin: center center !important;
        cursor: crosshair !important;
    }
    [data-testid="stPlotlyChart"]:hover {
        z-index: 8 !important;
        transform: scale(1.025) !important;
        filter: brightness(1.08) !important;
        box-shadow: 0 0 12px rgba(255,140,0,.92),
                    0 0 25px rgba(255,140,0,.68),
                    0 0 42px rgba(255,140,0,.32),
                    inset 0 0 17px rgba(255,140,0,.15) !important;
    }
    [data-testid="stPlotlyChart"] .js-plotly-plot { border-radius: 10px !important; }

    /* Аварийная карточка остаётся единственным крупным красным акцентом. */
    .emergency-alert {
        border: 1.5px solid #FF3B5C !important;
        box-shadow: 0 0 13px rgba(255,35,75,.58),
                    0 0 26px rgba(255,35,75,.30),
                    inset 0 0 17px rgba(255,35,75,.12) !important;
        animation: emergencyPulse 1.9s ease-in-out infinite !important;
    }
    @keyframes emergencyPulse {
        0%,100% { transform: scale(1); box-shadow: 0 0 11px rgba(255,35,75,.46), inset 0 0 12px rgba(255,35,75,.10); }
        50% { transform: scale(1.014); box-shadow: 0 0 20px rgba(255,35,75,.82), 0 0 34px rgba(255,35,75,.38), inset 0 0 18px rgba(255,35,75,.16); }
    }

    .st-key-archive-jump [data-testid="stButton"] button {
        background: linear-gradient(110deg,rgba(36,31,100,.98),rgba(20,32,74,.98)) !important;
        border: 1.5px solid #00F0FF !important;
        color: #E9FBFF !important;
        box-shadow: 0 0 12px rgba(0,240,255,.42), 0 0 20px rgba(79,70,229,.25) !important;
    }
    .st-key-archive-jump [data-testid="stButton"] button:hover {
        transform: scale(1.035) !important;
        box-shadow: 0 0 17px rgba(0,240,255,.78), 0 0 26px rgba(79,70,229,.42) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 4. SQLITE: ИСТОРИЯ ТЕЛЕМЕТРИИ, СОБЫТИЙ, РЕМОНТОВ И ОБОРУДОВАНИЯ
# =============================================================================


def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def db_session():
    connection = db_connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def timestamp_text(value: datetime | pd.Timestamp | date) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.max.time().replace(microsecond=0))
    return value.isoformat(timespec="seconds")


def initialize_database() -> None:
    """Создает локальную БД, заполняет год синтетических данных и архивирует старое."""
    with db_session() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                facility TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_a REAL NOT NULL,
                voltage_v REAL NOT NULL,
                temperature_c REAL NOT NULL,
                PRIMARY KEY (facility, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_telemetry_time
                ON telemetry (facility, timestamp);

            CREATE TABLE IF NOT EXISTS telemetry_archive (
                facility TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_a REAL NOT NULL,
                voltage_v REAL NOT NULL,
                temperature_c REAL NOT NULL,
                PRIMARY KEY (facility, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_telemetry_archive_time
                ON telemetry_archive (facility, timestamp);

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facility TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                parameter TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS repairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facility TEXT NOT NULL,
                repair_date TEXT NOT NULL,
                equipment_name TEXT NOT NULL,
                reason TEXT NOT NULL,
                work_done TEXT NOT NULL,
                engineer TEXT NOT NULL,
                result TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facility TEXT NOT NULL,
                equipment_name TEXT NOT NULL,
                manufacturer TEXT NOT NULL,
                model TEXT NOT NULL,
                serial_number TEXT NOT NULL,
                install_date TEXT NOT NULL,
                service_life_years INTEGER NOT NULL,
                manufacturer_contact TEXT NOT NULL,
                notes TEXT NOT NULL
            );
            """
        )

        total_rows = connection.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        if total_rows == 0:
            now_hour = pd.Timestamp.now().floor("h")
            date_index = pd.date_range(end=now_hour, periods=365 * 24, freq="h")
            seed_events: list[tuple[str, str, str, str, str, str]] = []

            for facility_index, facility in enumerate(FACILITIES):
                rng = np.random.default_rng(20261125 + facility_index * 173)
                phase = facility_index * 0.7
                t = np.linspace(0, 4 * np.pi, len(date_index))

                current = 15.0 + 1.25 * np.sin(t + phase) + rng.normal(0, 1.8, len(date_index))
                voltage = 380.0 + rng.normal(0, 4.0, len(date_index)) + 1.2 * np.sin(t / 2 + phase)
                temperature = 60.0 + 2.3 * np.sin(t + phase) + rng.normal(0, 2.4, len(date_index))

                # Условные диагностические эпизоды на историческом графике.
                episode_specs = [
                    (1150 + facility_index * 37, 5, 12.0, 8.0, -7.0),
                    (3900 + facility_index * 29, 4, 17.0, 23.0, -12.0),
                    (6900 - facility_index * 23, 6, 10.0, 14.0, -8.0),
                ]
                for start, length, current_add, temp_add, voltage_add in episode_specs:
                    end = min(start + length, len(date_index))
                    current[start:end] += current_add
                    temperature[start:end] += temp_add
                    voltage[start:end] += voltage_add

                rows = [
                    (
                        facility,
                        timestamp.to_pydatetime().isoformat(timespec="seconds"),
                        float(max(0, current_value)),
                        float(voltage_value),
                        float(max(0, temperature_value)),
                    )
                    for timestamp, current_value, voltage_value, temperature_value in zip(
                        date_index, current, voltage, temperature
                    )
                ]
                connection.executemany(
                    """INSERT OR IGNORE INTO telemetry
                       (facility, timestamp, current_a, voltage_v, temperature_c)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )

                for offset, parameter, description in [
                    (1150 + facility_index * 37, "Температура", "Учебный эпизод повышения температуры в исторических данных."),
                    (3900 + facility_index * 29, "Ток", "Учебный эпизод отклонения тока нагрузки."),
                    (6900 - facility_index * 23, "Напряжение", "Учебный эпизод отклонения напряжения сети."),
                ]:
                    event_time = date_index[min(offset, len(date_index) - 1)].to_pydatetime()
                    seed_events.append(
                        (
                            facility,
                            event_time.isoformat(timespec="seconds"),
                            "Историческое отклонение (симуляция)",
                            "Предупреждение",
                            parameter,
                            description,
                        )
                    )

            connection.executemany(
                """INSERT INTO events
                   (facility, timestamp, event_type, severity, parameter, description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                seed_events,
            )

        # В оперативной таблице остаются последние 365 дней. Более старые записи
        # автоматически переносятся в долговременный архив, а не удаляются.
        cutoff = timestamp_text(datetime.now() - timedelta(days=365))
        connection.execute(
            """INSERT OR IGNORE INTO telemetry_archive
               (facility, timestamp, current_a, voltage_v, temperature_c)
               SELECT facility, timestamp, current_a, voltage_v, temperature_c
               FROM telemetry WHERE timestamp < ?""",
            (cutoff,),
        )
        connection.execute("DELETE FROM telemetry WHERE timestamp < ?", (cutoff,))


def add_event(
    facility: str,
    event_type: str,
    severity: str,
    parameter: str,
    description: str,
) -> None:
    with db_session() as connection:
        connection.execute(
            """INSERT INTO events
               (facility, timestamp, event_type, severity, parameter, description)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (facility, timestamp_text(datetime.now()), event_type, severity, parameter, description),
        )


def append_measurement(facility: str, row: pd.Series) -> None:
    with db_session() as connection:
        connection.execute(
            """INSERT OR REPLACE INTO telemetry
               (facility, timestamp, current_a, voltage_v, temperature_c)
               VALUES (?, ?, ?, ?, ?)""",
            (
                facility,
                timestamp_text(pd.Timestamp(row["Время"])),
                float(row["Ток"]),
                float(row["Напряжение"]),
                float(row["Температура"]),
            ),
        )


def read_telemetry(
    facility: str,
    start: datetime,
    end: datetime,
    include_long_archive: bool = False,
) -> pd.DataFrame:
    start_text = timestamp_text(start)
    end_text = timestamp_text(end)
    source = "telemetry"
    if include_long_archive:
        source = """(
            SELECT facility, timestamp, current_a, voltage_v, temperature_c FROM telemetry
            UNION ALL
            SELECT facility, timestamp, current_a, voltage_v, temperature_c FROM telemetry_archive
        )"""
    query = f"""SELECT timestamp, current_a, voltage_v, temperature_c
                FROM {source}
                WHERE facility = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp"""
    with db_session() as connection:
        rows = connection.execute(query, (facility, start_text, end_text)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Время", "Ток", "Напряжение", "Температура"])
    data = pd.DataFrame([dict(row) for row in rows])
    data = data.rename(
        columns={
            "timestamp": "Время",
            "current_a": "Ток",
            "voltage_v": "Напряжение",
            "temperature_c": "Температура",
        }
    )
    data["Время"] = pd.to_datetime(data["Время"])
    return data


def read_events(facility: str | None = None, limit: int = 500) -> pd.DataFrame:
    query = "SELECT timestamp, facility, event_type, severity, parameter, description FROM events"
    params: list[Any] = []
    if facility:
        query += " WHERE facility = ?"
        params.append(facility)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(int(limit))
    with db_session() as connection:
        rows = connection.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Дата и время", "Объект", "Событие", "Уровень", "Параметр", "Описание"])
    frame = pd.DataFrame([dict(row) for row in rows])
    return frame.rename(
        columns={
            "timestamp": "Дата и время",
            "facility": "Объект",
            "event_type": "Событие",
            "severity": "Уровень",
            "parameter": "Параметр",
            "description": "Описание",
        }
    )


def read_repairs(facility: str | None = None) -> pd.DataFrame:
    query = """SELECT repair_date AS 'Дата ремонта', facility AS 'Объект',
               equipment_name AS 'Оборудование', reason AS 'Причина',
               work_done AS 'Выполненные работы', engineer AS 'Ответственный инженер',
               result AS 'Результат' FROM repairs"""
    params: list[Any] = []
    if facility:
        query += " WHERE facility = ?"
        params.append(facility)
    query += " ORDER BY repair_date DESC"
    with db_session() as connection:
        return pd.read_sql_query(query, connection, params=params)


def insert_repair(
    facility: str,
    repair_date: date,
    equipment_name: str,
    reason: str,
    work_done: str,
    engineer: str,
    result: str,
) -> None:
    with db_session() as connection:
        connection.execute(
            """INSERT INTO repairs
               (facility, repair_date, equipment_name, reason, work_done, engineer, result)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (facility, repair_date.isoformat(), equipment_name, reason, work_done, engineer, result),
        )


def read_equipment(facility: str | None = None) -> pd.DataFrame:
    query = """SELECT id AS 'ID', facility AS 'Объект', equipment_name AS 'Оборудование',
               manufacturer AS 'Производитель', model AS 'Модель',
               serial_number AS 'Заводской номер', install_date AS 'Дата установки',
               service_life_years AS 'Срок эксплуатации, лет',
               manufacturer_contact AS 'Контакты производителя', notes AS 'Примечание'
               FROM equipment"""
    params: list[Any] = []
    if facility:
        query += " WHERE facility = ?"
        params.append(facility)
    query += " ORDER BY id DESC"
    with db_session() as connection:
        return pd.read_sql_query(query, connection, params=params)


def insert_equipment(
    facility: str,
    equipment_name: str,
    manufacturer: str,
    model: str,
    serial_number: str,
    install_date: date,
    service_life_years: int,
    manufacturer_contact: str,
    notes: str,
) -> None:
    with db_session() as connection:
        connection.execute(
            """INSERT INTO equipment
               (facility, equipment_name, manufacturer, model, serial_number,
                install_date, service_life_years, manufacturer_contact, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                facility,
                equipment_name,
                manufacturer,
                model,
                serial_number,
                install_date.isoformat(),
                int(service_life_years),
                manufacturer_contact,
                notes,
            ),
        )


initialize_database()


# =============================================================================
# 5. ДАННЫЕ МОНИТОРИНГА И МОДЕЛЬ ИИ
# =============================================================================


def initialize_session_state() -> None:
    defaults = {
        "emergency": False,
        "override": False,
        "facility": FACILITIES[0],
        "monitoring_data": None,
        "monitoring_cache_key": None,
        "monitor_seed": 20261125,
        "expanded_parameter": None,
        "detail_parameter_choice": "Ток",
        "detail_period_mode": "Неделя",
        "main_page": "Мониторинг",
        "archive_section": "История измерений",
        "ignore_next_chart_selection": None,
        "chart_reset_token": 0,
        "simulator_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_monitoring_data(emergency: bool, seed: int) -> pd.DataFrame:
    """Создает 100 учебных точек телеметрии с минутным шагом."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("s"), periods=100, freq="min")
    current = rng.normal(loc=15.0, scale=2.0, size=100)
    voltage = rng.normal(loc=380.0, scale=5.0, size=100)
    temperature = rng.normal(loc=60.0, scale=3.0, size=100)

    if emergency:
        current[-10:] = rng.normal(loc=45.0, scale=5.0, size=10)
        temperature[-10:] = rng.normal(loc=110.0, scale=10.0, size=10)

    return pd.DataFrame(
        {
            "Время": timestamps,
            "Ток": np.clip(current, 0, None),
            "Напряжение": voltage,
            "Температура": np.clip(temperature, 0, None),
        }
    )


def get_monitoring_data() -> pd.DataFrame:
    cache_key = (bool(st.session_state.emergency), st.session_state.facility, int(st.session_state.monitor_seed))
    if st.session_state.monitoring_cache_key != cache_key or st.session_state.monitoring_data is None:
        facility_index = FACILITIES.index(st.session_state.facility)
        seed = int(st.session_state.monitor_seed + facility_index * 101 + int(st.session_state.emergency))
        st.session_state.monitoring_data = create_monitoring_data(bool(st.session_state.emergency), seed)
        st.session_state.monitoring_cache_key = cache_key
    return st.session_state.monitoring_data.copy()


def train_anomaly_model(seed: int = 42) -> IsolationForest:
    rng = np.random.default_rng(seed)
    normal_samples = pd.DataFrame(
        {
            "Ток": rng.normal(15.0, 2.0, 900),
            "Напряжение": rng.normal(380.0, 5.0, 900),
            "Температура": rng.normal(60.0, 3.0, 900),
        }
    )
    model = IsolationForest(n_estimators=160, contamination=0.025, random_state=seed)
    model.fit(normal_samples)
    return model


@st.cache_resource(show_spinner=False)
def get_ai_model() -> IsolationForest:
    return train_anomaly_model()


def classify_last_point(data: pd.DataFrame) -> tuple[str, float]:
    model = get_ai_model()
    point = data[["Ток", "Напряжение", "Температура"]].tail(1)
    prediction = int(model.predict(point)[0])
    score = float(model.decision_function(point)[0])
    return ("аномалия" if prediction == -1 else "норма", score)


def is_parameter_anomaly(data: pd.Series, parameter: str) -> pd.Series:
    values = data[parameter].astype(float)
    if parameter == "Ток":
        return (values > CURRENT_CRITICAL_A) | (values < 5)
    if parameter == "Напряжение":
        return (values < VOLTAGE_LOW_V) | (values > VOLTAGE_HIGH_V)
    return values >= TEMPERATURE_CRITICAL_C


initialize_session_state()


# =============================================================================
# 6. ГРАФИКИ И HTML-ТАБЛИЦЫ
# =============================================================================


def add_reference_zones(fig: go.Figure, parameter: str) -> None:
    meta = PARAMETERS[parameter]
    if parameter == "Ток":
        fig.add_hrect(y0=10, y1=20, fillcolor="rgba(79,70,229,0.10)", line_width=0, layer="below")
        fig.add_hrect(y0=20, y1=40, fillcolor="rgba(255,140,0,0.09)", line_width=0, layer="below")
        fig.add_hrect(y0=40, y1=65, fillcolor="rgba(255,140,0,0.14)", line_width=0, layer="below")
        fig.add_hline(y=10, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=20, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=40, line_dash="dash", line_color="#FF8C00", opacity=0.95)
    elif parameter == "Напряжение":
        fig.add_hrect(y0=370, y1=390, fillcolor="rgba(79,70,229,0.11)", line_width=0, layer="below")
        fig.add_hrect(y0=355, y1=370, fillcolor="rgba(255,140,0,0.09)", line_width=0, layer="below")
        fig.add_hrect(y0=390, y1=405, fillcolor="rgba(255,140,0,0.09)", line_width=0, layer="below")
        fig.add_hrect(y0=340, y1=355, fillcolor="rgba(255,140,0,0.14)", line_width=0, layer="below")
        fig.add_hrect(y0=405, y1=420, fillcolor="rgba(255,140,0,0.14)", line_width=0, layer="below")
        fig.add_hline(y=370, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=390, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=380, line_dash="dash", line_color="#00F0FF", opacity=0.8)
    else:
        fig.add_hrect(y0=50, y1=70, fillcolor="rgba(79,70,229,0.11)", line_width=0, layer="below")
        fig.add_hrect(y0=70, y1=90, fillcolor="rgba(255,140,0,0.10)", line_width=0, layer="below")
        fig.add_hrect(y0=90, y1=130, fillcolor="rgba(255,140,0,0.15)", line_width=0, layer="below")
        fig.add_hline(y=50, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=70, line_dash="dot", line_color="#4F46E5", opacity=0.9)
        fig.add_hline(y=90, line_dash="dash", line_color="#FF8C00", opacity=0.95)


def make_overview_chart(data: pd.DataFrame, parameter: str) -> go.Figure:
    meta = PARAMETERS[parameter]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Время"],
            y=data[parameter],
            mode="lines+markers",
            name=f"{parameter}, {meta['unit']}",
            line={"color": meta["color"], "width": 2.7},
            marker={"size": 6, "color": meta["color"], "opacity": 0.82, "line": {"width": 0.6, "color": "#F8FAFC"}},
            hovertemplate="%{x|%d.%m.%Y %H:%M:%S}<br>"
            + f"{parameter}: %{{y:.2f}} {meta['unit']}<extra></extra>",
        )
    )
    average = float(data[parameter].mean())
    fig.add_hline(
        y=average,
        line_dash="dash",
        line_color="#FF8C00",
        line_width=1.5,
        annotation_text=f"Среднее {average:.1f} {meta['unit']}",
        annotation_position="top left",
        annotation_font_color="#FFB454",
        annotation_font_size=9,
    )
    add_reference_zones(fig, parameter)
    fig.update_layout(
        title={"text": meta["description"], "x": 0.035, "xanchor": "left", "font": {"size": 14, "color": "#F8FAFC"}},
        height=355,
        margin={"l": 16, "r": 13, "t": 53, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,10,20,0.62)",
        font={"family": "Montserrat, sans-serif", "color": "#CBD5E1", "size": 10},
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        title="Дата и время",
        tickformat="%H:%M\n%d.%m",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.13)",
        linecolor="rgba(148,163,184,0.28)",
        tickfont={"color": "#A5B4FC", "size": 9},
    )
    fig.update_yaxes(
        title=f"{parameter}, {meta['unit']}",
        range=meta["range"],
        showgrid=True,
        gridcolor="rgba(148,163,184,0.13)",
        linecolor="rgba(148,163,184,0.28)",
        tickfont={"color": "#CBD5E1", "size": 9},
    )
    return fig


def make_history_chart(data: pd.DataFrame, parameter: str, title_suffix: str = "") -> go.Figure:
    meta = PARAMETERS[parameter]
    fig = go.Figure()
    if data.empty:
        fig.update_layout(
            title="Нет измерений за выбранный период",
            height=510,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(5,10,20,0.62)",
            font={"family": "Montserrat, sans-serif", "color": "#CBD5E1"},
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=data["Время"],
            y=data[parameter],
            mode="lines",
            name=f"{parameter}, {meta['unit']}",
            line={"color": meta["color"], "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(0,240,255,0.035)" if parameter == "Ток" else "rgba(79,70,229,0.045)",
            hovertemplate="%{x|%d.%m.%Y %H:%M}<br>"
            + f"{parameter}: %{{y:.2f}} {meta['unit']}<extra></extra>",
        )
    )
    average = float(data[parameter].mean())
    fig.add_hline(
        y=average,
        line_dash="dash",
        line_color="#FF8C00",
        line_width=1.8,
        annotation_text=f"Среднее: {average:.2f} {meta['unit']}",
        annotation_position="top left",
        annotation_font_color="#FFB454",
    )
    add_reference_zones(fig, parameter)

    anomaly_mask = is_parameter_anomaly(data, parameter)
    anomaly_data = data.loc[anomaly_mask]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data["Время"],
                y=anomaly_data[parameter],
                mode="markers",
                name="Аномальные точки",
                marker={"color": "#FF8C00", "size": 6, "symbol": "circle-open", "line": {"width": 1.6}},
                hovertemplate="Отклонение<br>%{x|%d.%m.%Y %H:%M}<br>%{y:.2f} " + meta["unit"] + "<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": f"{meta['description']} · {title_suffix}" if title_suffix else meta["description"],
            "x": 0.025,
            "xanchor": "left",
            "font": {"size": 16, "color": "#F8FAFC"},
        },
        height=520,
        margin={"l": 25, "r": 20, "t": 60, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,10,20,0.62)",
        font={"family": "Montserrat, sans-serif", "color": "#CBD5E1", "size": 11},
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right", "font": {"size": 10}},
    )
    fig.update_xaxes(
        title="Дата и время измерения",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.13)",
        tickformat="%d.%m.%Y<br>%H:%M",
        rangeslider={
            "visible": True,
            "bgcolor": "rgba(79,70,229,0.19)",
            "bordercolor": "#00BFFF",
            "borderwidth": 1,
            "thickness": 0.12,
        },
    )
    fig.update_yaxes(
        title=f"{parameter}, {meta['unit']}",
        range=meta["range"],
        showgrid=True,
        gridcolor="rgba(148,163,184,0.13)",
    )
    return fig


def make_html_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    return (
        '<div class="docs-table-wrap"><table class="requirements-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody></table></div>"
    )


def safe_cell(value: str, strong: bool = False) -> str:
    cleaned = html.escape(value)
    return f"<strong>{cleaned}</strong>" if strong else cleaned


# =============================================================================
# 7. СОДЕРЖАНИЕ ДОКУМЕНТАЦИИ — ПО СТАТЬЕ
# =============================================================================

FUNCTIONAL_REQUIREMENTS = [
    (
        "F-01",
        "Контроль целостности изоляции",
        "Система должна анализировать емкостные изменения (частичные разряды) в контрольных слоях многослойной изоляции и фиксировать их исчезновение для раннего обнаружения повреждения внутренних слоев.",
    ),
    (
        "F-02",
        "Прогнозирование остаточного ресурса",
        "ИИ-модуль должен анализировать скорость изменения параметров (температуры, состояния изоляции) и прогнозировать остаточный ресурс оборудования, формируя трехуровневые статусы оповещения: желтый — замену необходимо провести в течение 60 дней; оранжевый — в течение 30 дней; красный — требуется немедленное вмешательство.",
    ),
    (
        "F-03",
        "Контроль теплового состояния контактов",
        "Система должна фиксировать отклонения температуры контактных соединений от эталонных значений. При критическом отклонении ИИ должен формировать сигнал о риске короткого замыкания.",
    ),
    (
        "F-04",
        "Динамическая балансировка нагрузки",
        "ИИ-модуль должен анализировать номиналы подключенных потребителей. При перегрузке система должна рассчитывать альтернативные маршруты и автоматически перенаправлять потоки электроэнергии по свободным ветвям сети.",
    ),
    (
        "F-05",
        "Дифференциальная защита от утечек",
        "Контроллеры должны непрерывно сравнивать входящие и выходящие значения токов. При фиксации паразитных токов (включая контур заземления и смежные компоненты) система должна автоматически изолировать аварийный участок.",
    ),
    (
        "F-06",
        "Прогнозирование микро-утечек",
        "ИИ-модуль должен анализировать динамику изменения микро-утечек и уведомлять диспетчера о расчетном времени срабатывания аппаратной защиты для планирования ремонтных работ.",
    ),
    (
        "F-07",
        "Защита от механических воздействий",
        "При фиксации механического воздействия (обрыв, падение посторонних предметов) система должна ограничивать подачу тока или перенаправлять его по резервным (подземным) линиям.",
    ),
    (
        "F-08",
        "Бесконтактная диагностика",
        "Система должна обрабатывать данные от портативных диагностических комплексов (сканеров) сотрудников для бесконтактного считывания параметров теплового/магнитного поля и формирования заключения о техническом состоянии оборудования.",
    ),
    (
        "F-09",
        "Взаимное резервирование датчиков",
        "Датчики должны передавать данные смежным узлам. При выходе из строя одного элемента соседние датчики должны фиксировать аномалии и передавать сигнал тревоги.",
    ),
    (
        "F-10",
        "Мониторинг биообъектов (животных)",
        "Система должна собирать данные о физиологических показателях животных (температура тела, двигательная активность) с помощью носимых устройств и сравнивать их с индивидуальной нормой.",
    ),
    (
        "F-11",
        "Анализ состояния поголовья",
        "ИИ-модуль при выявлении отклонений физиологических показателей у большинства животных должен формировать сигнал об аварии в системе жизнеобеспечения (отключение вентиляции/отопления). При отклонениях у единиц — продолжать мониторинг.",
    ),
    (
        "F-12",
        "Управление резервным энергоснабжением",
        "При критической аварии ИИ-система должна анализировать данные датчиков окружающей среды (анемометр, люксметр, расходомер) и прогноз погоды. На основе анализа система должна подключать наиболее эффективный альтернативный источник энергии (ветрогенератор, солнечные панели, микроГЭС или биогазовую установку) для поддержания систем жизнеобеспечения АПК.",
    ),
]

NON_FUNCTIONAL_REQUIREMENTS = [
    ("NF-01", "Быстродействие", "Время отклика системы при расчете аномалии и выявлении утечки не должно превышать 3 секунд."),
    ("NF-02", "Надежность (доступность)", "Система должна быть доступна 99,9% времени (без учета плановых технических работ)."),
    ("NF-03", "Безопасность", "Доступ к системе и управлению путями должен осуществляться строго через авторизацию (по ролям: диспетчер, инженер)."),
    ("NF-04", "Масштабируемость", "Система должна поддерживать подключение новых датчиков (например, добавление носимых устройств на новых животных или установка новых сканеров) без потери производительности."),
    ("NF-05", "Отказоустойчивость", "При выходе из строя одного из датчиков система должна сохранять работоспособность за счет дублирования данных с соседних узлов."),
]


# =============================================================================
# 8. ДИАЛОГОВЫЕ ОКНА
# =============================================================================

@st.dialog("Меню платформы", width="medium")
def show_menu_dialog() -> None:
    st.markdown(
        """
        <div class="reference-section">
            <h4>Разделы платформы</h4>
            <p><strong>Мониторинг</strong> — текущие измерения тока, напряжения и температуры, графики и анализ аномалий.</p>
            <p><strong>Симулятор</strong> — проверка выбранных параметров с помощью демонстрационной модели ИИ.</p>
            <p><strong>Архив и обслуживание</strong> — история измерений, ремонтных работ, оборудования и событий.</p>
            <p><strong>Документация</strong> — системные требования, роли пользователей, термины и правила платформы.</p>
        </div>
        <div class="reference-section">
            <h4>Логика архитектуры</h4>
            <p>Сбор данных IoT → передача данных Pipeline → обработка AI Core → интерфейс диспетчера UI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Об авторе", width=540)
def show_author_dialog() -> None:
    st.html(
        textwrap.dedent(f"""
        <div class="author-dialog-panel">
            <div class="author-kicker">ПРОЕКТ · МОНИТОРИНГ АПК</div>
            <h2>Платформа интеллектуального мониторинга электрических подстанций</h2>

            <h3>О проекте</h3>
            <p class="author-intro">Это демонстрационная веб-платформа для наблюдения за состоянием электрических подстанций агропромышленного комплекса.</p>
            <p>Сайт отображает параметры тока, напряжения и температуры, строит графики измерений, показывает события и позволяет исследовать возможные отклонения с помощью аналитических методов и алгоритмов искусственного интеллекта.</p>

            <h3>Актуальность и польза</h3>
            <p>Надёжное электроснабжение важно для бесперебойной работы ферм, элеваторов, птицефабрик и другого оборудования АПК. Своевременное обнаружение необычных изменений помогает обратить внимание на потенциальные неисправности, оценить состояние оборудования и планировать техническое обслуживание заранее.</p>
            <p class="author-note">Платформа предназначена для демонстрации возможностей мониторинга и анализа данных. Показания и аварийные сценарии являются программной симуляцией; сайт не подключён к промышленным датчикам и не управляет реальной электроустановкой.</p>

            <h3>Благодарность</h3>
            <p>Благодарю за интерес к проекту и внимание к моей работе. Надеюсь, платформа поможет наглядно представить, как цифровые технологии и анализ данных могут способствовать повышению надёжности электроснабжения объектов агропромышленного комплекса.</p>

            <div class="author-footer">
                <span>АВТОР ПРОЕКТА</span>
                <strong>Болтикова Софья Александровна</strong>
                <small><a href="https://github.com/esoffias" target="_blank" rel="noopener noreferrer">GitHub · github.com/esoffias</a></small>
            </div>
        </div>
        """).strip(),
    )

@st.dialog("Функциональные требования · F-01—F-12", width="large")
def show_functional_requirements() -> None:
    rows = [
        (safe_cell(req_id, True), safe_cell(name), safe_cell(description))
        for req_id, name, description in FUNCTIONAL_REQUIREMENTS
    ]
    st.markdown(
        "<p style='color:#64748B;font-size:0.78rem;'>Функциональные требования описывают действия, которые должна выполнять проектируемая система.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(make_html_table(["ID", "Название", "Описание требования"], rows), unsafe_allow_html=True)


@st.dialog("Нефункциональные требования · NF-01—NF-05", width="large")
def show_nonfunctional_requirements() -> None:
    rows = [
        (safe_cell(req_id, True), safe_cell(name), safe_cell(description))
        for req_id, name, description in NON_FUNCTIONAL_REQUIREMENTS
    ]
    st.markdown(
        "<p style='color:#64748B;font-size:0.78rem;'>Нефункциональные требования описывают показатели качества и условия работы системы.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(make_html_table(["ID", "Название", "Описание требования"], rows), unsafe_allow_html=True)


@st.dialog("Роли пользователей и права доступа", width="large")
def show_roles_dialog() -> None:
    roles = [
        (
            safe_cell("Диспетчер", True),
            safe_cell("Контролирует показатели и графики; принимает уведомления; анализирует состояние узлов; координирует передачу информации ремонтной бригаде."),
            safe_cell("Просмотр телеметрии, графиков, журнала событий и рекомендаций; регистрация и подтверждение уведомлений."),
        ),
        (
            safe_cell("Инженер", True),
            safe_cell("Проводит техническую диагностику и обслуживание; проверяет выявленные отклонения; фиксирует результаты осмотра и ремонта оборудования."),
            safe_cell("Просмотр технических параметров и диагностических данных; внесение результатов обслуживания и ремонта в журнал."),
        ),
    ]
    st.markdown(
        make_html_table(["Роль", "Обязанности", "Предусмотренные задачи"], roles),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="docs-table-wrap" style="margin-top:1rem;padding:1rem;max-height:none;">
            <h4 style="color:#C8C1FF;margin-top:0;">Ограничения и правила использования</h4>
            <ul class="role-restrictions">
                <li>Демонстрационный интерфейс не выполняет физические переключения и не управляет реальным оборудованием.</li>
                <li>Нельзя использовать симуляционные показатели как подтвержденные измерения подстанции.</li>
                <li>Нельзя изменять реальные уставки релейной защиты или обходить аппаратные блокировки через этот прототип.</li>
                <li>Записи о ремонте и замене оборудования должны вноситься с проверкой фактических паспортных данных.</li>
                <li>Любые реальные работы на электроустановке выполняются только уполномоченным персоналом по регламентам организации.</li>
            </ul>
            <p style="color:#AAB7D1;font-size:0.74rem;">Роли «диспетчер» и «инженер» указаны в требовании NF-03. Детализация задач и ограничений является проектным описанием интерфейса.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Термины и правила платформы", width="large")
def show_glossary_dialog() -> None:
    glossary_sections = [
        (
            "Основные понятия",
            """
            <p><strong>Электрическая подстанция</strong> — электроустановка, предназначенная для приема, преобразования и распределения электрической энергии.</p>
            <p><strong>АПК</strong> — агропромышленный комплекс; в рамках проекта рассматриваются фермы, элеваторы, птицефабрики и связанные объекты.</p>
            <p><strong>Предиктивное обслуживание</strong> — подход к техническому обслуживанию, при котором решения о проверке и ремонте принимаются на основе текущего состояния оборудования и тенденций изменения его параметров.</p>
            <p><strong>Требование системы</strong> — сформулированное условие или функция, которую должна обеспечивать проектируемая информационная система.</p>
            """,
        ),
        (
            "Электротехнические параметры",
            """
            <p><strong>Ток (I)</strong> — физическая величина, характеризующая направленное движение электрического заряда. В интерфейсе отображается в амперах (А).</p>
            <p><strong>Напряжение (U)</strong> — разность электрических потенциалов между двумя точками цепи. В интерфейсе отображается в вольтах (В).</p>
            <p><strong>Температура (T)</strong> — показатель теплового состояния оборудования. В интерфейсе отображается в градусах Цельсия (°C).</p>
            <p><strong>Перегрузка</strong> — режим, при котором ток или нагрузка превышает допустимый для конкретного оборудования уровень.</p>
            <p><strong>Короткое замыкание</strong> — аварийное соединение точек электрической цепи с различными потенциалами через малое сопротивление.</p>
            <p><strong>Утечка тока</strong> — прохождение части тока по нежелательному пути, например через поврежденную изоляцию или на землю.</p>
            """,
        ),
        (
            "Архитектура и искусственный интеллект",
            """
            <p><strong>IoT</strong> — слой датчиков и контроллеров, собирающих первичные данные об оборудовании и окружающей среде.</p>
            <p><strong>Pipeline</strong> — уровень передачи собранных данных к серверу и модулю обработки.</p>
            <p><strong>AI Core</strong> — модуль алгоритмов и математических моделей для анализа показателей, обнаружения аномалий и формирования прогнозов.</p>
            <p><strong>UI</strong> — пользовательский интерфейс, через который диспетчер просматривает графики, статусы и уведомления.</p>
            <p><strong>IsolationForest</strong> — алгоритм машинного обучения, который выявляет точки, статистически отличающиеся от основной массы наблюдений.</p>
            <p><strong>Аномалия</strong> — нетипичное значение или сочетание значений, которое требует дополнительной проверки.</p>
            """,
        ),
        (
            "Правила пользования платформой",
            """
            <p>1. Сначала выберите объект АПК, для которого требуется просмотреть показатели.</p>
            <p>2. Сравнивайте текущие значения с графиками, средним уровнем и отображенными ориентировочными диапазонами.</p>
            <p>3. При просмотре истории проверяйте дату и время каждой точки измерения.</p>
            <p>4. Используйте журнал событий для сопоставления отклонений с предупреждениями и действиями пользователя.</p>
            <p>5. Отделяйте демонстрационные данные от фактических измерений. В текущем прототипе показатели генерируются программно.</p>
            <p>6. Записи об оборудовании, производителе, контактах и сроках службы заполняйте по паспортам и документам конкретного оборудования.</p>
            """,
        ),
        (
            "Техника безопасности",
            """
            <p>Платформа предназначена для ознакомления с концепцией мониторинга и не заменяет эксплуатационную документацию, инструкции предприятия, нарядно-допускную систему или работу устройств релейной защиты.</p>
            <p>Не используйте учебные значения и демонстрационные пороги для выбора режимов работы реальной электроустановки.</p>
            <p>Не выполняйте переключения, ремонт, измерения или приближение к токоведущим частям на основании подсказок этого интерфейса.</p>
            <p>Работы на электроустановках выполняет только уполномоченный персонал с необходимой подготовкой, допуском и средствами защиты в соответствии с действующими правилами организации.</p>
            """,
        ),
    ]

    for title, body in glossary_sections:
        st.markdown(
            f'<div class="reference-section"><h4>{html.escape(title)}</h4>{body}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# 9. ВЕРХНЯЯ ПАНЕЛЬ
# =============================================================================


def navigate_to_archive_history() -> None:
    """Переключает основной раздел и сразу открывает историю измерений."""
    st.session_state.main_page = "Архив и обслуживание"
    st.session_state.archive_section = "История измерений"
    st.session_state.expanded_parameter = None


nav_left, nav_center, nav_right = st.columns([1.25, 5.5, 1.4], vertical_alignment="center")
with nav_left:
    if st.button("☰  Меню", key="open_main_menu", width="stretch"):
        show_menu_dialog()
with nav_center:
    st.markdown(f'<div class="hero-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Интеллектуальная система контроля технического состояния</div>', unsafe_allow_html=True)
with nav_right:
    if st.button("♙  Об авторе", key="open_author_info", width="stretch"):
        show_author_dialog()

st.markdown(
    '<div style="text-align:center;margin:-0.1rem 0 1.0rem;"><span class="status-pill"><span class="status-dot"></span> ДЕМОНСТРАЦИОННАЯ СРЕДА · ДАННЫЕ СИМУЛИРУЮТСЯ</span></div>',
    unsafe_allow_html=True,
)


main_page = st.radio(
    "Основные разделы",
    ["Мониторинг", "Симулятор", "Архив и обслуживание", "Документация"],
    horizontal=True,
    key="main_page",
    label_visibility="collapsed",
)


# =============================================================================
# 10. ВКЛАДКА «МОНИТОРИНГ»
# =============================================================================

if main_page == "Мониторинг":
    st.markdown('<div class="section-heading">Панель текущего состояния</div>', unsafe_allow_html=True)
    control_row = st.columns([1.8, 1.0, 1.0, 1.0], vertical_alignment="bottom")
    with control_row[0]:
        selected_facility = st.selectbox(
            "Контролируемый объект",
            FACILITIES,
            index=FACILITIES.index(st.session_state.facility),
            key="monitor_facility_select",
        )
        if selected_facility != st.session_state.facility:
            st.session_state.facility = selected_facility
            st.session_state.monitoring_data = None
            st.session_state.monitoring_cache_key = None
            st.session_state.expanded_parameter = None
            st.rerun()
    with control_row[1]:
        if st.button("Симулировать аварию", key="trigger_emergency", width="stretch"):
            st.session_state.emergency = True
            st.session_state.override = False
            st.session_state.monitor_seed += 1
            st.session_state.monitoring_data = None
            st.session_state.monitoring_cache_key = None
            add_event(
                st.session_state.facility,
                "Аварийный сценарий запущен",
                "Критический",
                "Ток / температура",
                "В учебной симуляции смоделирован рост тока и температуры на последних десяти точках.",
            )
            st.rerun()
    with control_row[2]:
        if st.button("Сбросить сценарий", key="reset_emergency", width="stretch"):
            was_active = bool(st.session_state.emergency)
            st.session_state.emergency = False
            st.session_state.override = False
            st.session_state.monitor_seed += 1
            st.session_state.monitoring_data = None
            st.session_state.monitoring_cache_key = None
            if was_active:
                add_event(
                    st.session_state.facility,
                    "Аварийный сценарий сброшен",
                    "Информация",
                    "—",
                    "Пользователь вернул демонстрацию в условный нормальный режим.",
                )
            st.rerun()
    with control_row[3]:
        if st.button("Обновить показания", key="refresh_measurements", width="stretch"):
            st.session_state.monitor_seed += 1
            st.session_state.monitoring_data = None
            st.session_state.monitoring_cache_key = None
            latest_frame = get_monitoring_data()
            append_measurement(st.session_state.facility, latest_frame.iloc[-1])
            add_event(
                st.session_state.facility,
                "Обновление телеметрии",
                "Информация",
                "Ток / напряжение / температура",
                "В локальную демонстрационную базу добавлена новая точка телеметрии.",
            )
            st.rerun()

    monitoring_data = get_monitoring_data()
    ai_class, ai_score = classify_last_point(monitoring_data)
    latest = monitoring_data.iloc[-1]

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Ток нагрузки", f"{latest['Ток']:.1f} А", "аварийный сценарий" if st.session_state.emergency else "текущий профиль", delta_color="off")
    metric_cols[1].metric("Напряжение сети", f"{latest['Напряжение']:.1f} В", "условное измерение", delta_color="off")
    metric_cols[2].metric("Температура", f"{latest['Температура']:.1f} °C", "перегрев" if latest["Температура"] >= TEMPERATURE_CRITICAL_C else "текущее значение", delta_color="off")
    metric_cols[3].metric("AI Core", "АНОМАЛИЯ" if ai_class == "аномалия" or st.session_state.emergency else "НОРМА", f"оценка {ai_score:.3f}", delta_color="off")

    st.markdown('<div class="section-heading" style="margin-top:1.35rem;">Графики электротехнических параметров</div>', unsafe_allow_html=True)
    expanded_parameter = st.session_state.expanded_parameter

    if expanded_parameter is None:
        chart_cols = st.columns(3, gap="medium")
        for column, parameter in zip(chart_cols, PARAMETERS.keys()):
            with column:
                chart_event = st.plotly_chart(
                    make_overview_chart(monitoring_data, parameter),
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": True},
                    key=f"overview_chart_{parameter}_{st.session_state.chart_reset_token}",
                    on_select="rerun",
                    selection_mode="points",
                )
                try:
                    selected_points = chart_event.selection.point_indices
                except (AttributeError, TypeError):
                    selected_points = []
                ignore_selection = st.session_state.ignore_next_chart_selection == parameter
                if ignore_selection:
                    st.session_state.ignore_next_chart_selection = None
                elif selected_points:
                    st.session_state.expanded_parameter = parameter
                    st.session_state.detail_parameter_choice = parameter
                    st.session_state.detail_parameter_selectbox = parameter
                    st.rerun()
                st.caption("Нажмите на точку графика или кнопку ниже, чтобы открыть подробный анализ.")
                if st.button(
                    f"Подробнее и история · {PARAMETERS[parameter]['symbol']}",
                    key=f"open_detail_{parameter}",
                    width="stretch",
                ):
                    st.session_state.expanded_parameter = parameter
                    st.session_state.detail_parameter_choice = parameter
                    st.session_state.detail_parameter_selectbox = parameter
                    st.rerun()
    else:
        st.markdown(
            f'<div class="section-heading">Подробный анализ параметра · {html.escape(st.session_state.expanded_parameter)}</div>',
            unsafe_allow_html=True,
        )
        detail_controls = st.columns([1.25, 1.35, 2.4, 1.0], vertical_alignment="bottom", gap="medium")
        with detail_controls[0]:
            selected_detail = st.selectbox(
                "Показатель",
                list(PARAMETERS.keys()),
                index=list(PARAMETERS.keys()).index(st.session_state.detail_parameter_choice),
                key="detail_parameter_selectbox",
            )
            st.session_state.detail_parameter_choice = selected_detail
        with detail_controls[1]:
            period_mode = st.radio(
                "Шаг просмотра",
                ["День", "Неделя", "Месяц"],
                horizontal=True,
                key="detail_period_mode",
            )
        with detail_controls[2]:
            cursor_date = st.slider(
                "Перемещение по истории (последние 365 дней)",
                min_value=date.today() - timedelta(days=364),
                max_value=date.today(),
                value=date.today(),
                step=timedelta(days=1),
                format="DD.MM.YYYY",
                key="detail_history_cursor",
            )
        with detail_controls[3]:
            if st.button("← К обзору", key="close_detail", width="stretch"):
                st.session_state.ignore_next_chart_selection = st.session_state.expanded_parameter
                st.session_state.chart_reset_token += 1
                st.session_state.expanded_parameter = None
                st.rerun()

        period_days = {"День": 1, "Неделя": 7, "Месяц": 30}[period_mode]
        period_end = datetime.combine(cursor_date, datetime.max.time().replace(microsecond=0))
        period_start = datetime.combine(cursor_date - timedelta(days=period_days - 1), datetime.min.time())
        year_limit = datetime.combine(date.today() - timedelta(days=364), datetime.min.time())
        if period_start < year_limit:
            period_start = year_limit
        detail_data = read_telemetry(
            st.session_state.facility,
            period_start,
            period_end,
            include_long_archive=False,
        )
        meta = PARAMETERS[selected_detail]
        st.markdown(
            f'<div class="glass-card"><span class="parameter-chip">{meta["symbol"]}</span><strong style="color:#F8FAFC;">{html.escape(selected_detail)} · {meta["unit"]}</strong><p style="margin-top:0.6rem;">Период: {period_start:%d.%m.%Y %H:%M} — {period_end:%d.%m.%Y %H:%M}. Для просмотра данных старше года используйте архив измерений.</p></div>',
            unsafe_allow_html=True,
        )
        detail_figure = make_history_chart(detail_data, selected_detail, f"{period_mode.lower()} · подробный просмотр")
        detail_figure.update_layout(height=650, margin={"l": 34, "r": 28, "t": 64, "b": 50})
        st.plotly_chart(
            detail_figure,
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
            key="expanded_parameter_history_chart",
        )

        if not detail_data.empty:
            anomaly_mask = is_parameter_anomaly(detail_data, selected_detail)
            values = detail_data[selected_detail]
            stat_cols = st.columns(5, gap="medium")
            stat_cols[0].metric("Измерений", f"{len(values):,}".replace(",", " "))
            stat_cols[1].metric("Минимум", f"{values.min():.2f} {meta['unit']}")
            stat_cols[2].metric("Среднее", f"{values.mean():.2f} {meta['unit']}")
            stat_cols[3].metric("Максимум", f"{values.max():.2f} {meta['unit']}")
            stat_cols[4].metric("Отклонений", int(anomaly_mask.sum()))
            anomaly_rows = detail_data.loc[anomaly_mask].tail(15)
            with st.expander("Подробный список точек с отклонениями", expanded=False):
                if anomaly_rows.empty:
                    st.caption("За выбранный период пороговые отклонения не обнаружены.")
                else:
                    view = anomaly_rows[["Время", selected_detail]].copy()
                    view["Время"] = view["Время"].dt.strftime("%d.%m.%Y %H:%M:%S")
                    view = view.rename(columns={"Время": "Дата и время", selected_detail: f"{selected_detail}, {meta['unit']}"})
                    st.dataframe(view, hide_index=True, width="stretch", height=240)
        else:
            st.info("За выбранный период сохранённых точек нет. Попробуйте другую дату или период.")

        st.caption("Пунктирная оранжевая линия — среднее значение. Индиго — ориентировочный рабочий диапазон. Оранжевые маркеры — точки за пределами демонстрационных порогов.")
        with st.container(key="archive-jump"):
            st.button(
                "▣  Архив · история измерений",
                key="open_archive_from_detail",
                width="content",
                on_click=navigate_to_archive_history,
            )

    st.markdown('<div class="section-heading" style="margin-top:1.45rem;">Состояние системы и действия</div>', unsafe_allow_html=True)
    status_col, action_col = st.columns([1.12, 1.0], gap="large")
    with status_col:
        st.markdown('<div class="section-heading">Статус ИИ-анализа</div>', unsafe_allow_html=True)
        if st.session_state.emergency:
            st.markdown(
                """
                <div class="emergency-alert">
                    <strong>КРИТИЧЕСКАЯ СИТУАЦИЯ · АВАРИЙНЫЙ СЦЕНАРИЙ</strong>
                    <p>Смоделированы повышенные значения тока и температуры. Требуется внимание диспетчера.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="glass-card">
                    <h4>Выявленное отклонение</h4>
                    <p>Последние десять точек телеметрии содержат искусственно заданный рост тока и температуры.</p>
                    <p><strong>Рекомендуемая логика концепции:</strong> уведомить диспетчера, проверить состояние контактов и изоляции, сформировать задачу для инженерной службы.</p>
                    <p style="color:#FCA5A5;">Это учебный сценарий, а не реальная оценка вероятности аварии или взрыва.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif ai_class == "аномалия":
            st.warning("ИИ-модель отметила последнее сочетание параметров как статистически нетипичное. Требуется дополнительная проверка.")
            st.caption("IsolationForest сравнивает измерение с синтетическим условно-нормальным профилем.")
        else:
            st.success("Показатели находятся в пределах текущего демонстрационного профиля.")
            st.caption("Последняя точка не классифицирована IsolationForest как аномальная.")

    with action_col:
        st.markdown('<div class="section-heading">Панель действий</div>', unsafe_allow_html=True)
        if st.session_state.emergency and not st.session_state.override:
            st.markdown(
                """
                <div class="glass-card">
                    <h4>Автоматическая реакция · симуляция</h4>
                    <p>В рамках концепции система формирует уведомление и рекомендует проверить аварийный участок и доступность резервного маршрута.</p>
                    <p style="color:#FCA5A5;">Физические выключатели и реальные линии не затрагиваются.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("ОТМЕНИТЬ АВТО-ДЕЙСТВИЕ", key="override_action", width="stretch"):
                st.session_state.override = True
                add_event(st.session_state.facility, "Автоматический сценарий приостановлен", "Информация", "—", "Пользователь включил безопасный демонстрационный режим.")
                st.rerun()
        elif st.session_state.override:
            st.info("Режим: Безопасный минимум")
            st.markdown(
                '<div class="glass-card"><p>Автоматический сценарий приостановлен пользователем. Учебные значения продолжают отображаться.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Снять блокировку", key="release_override", width="stretch"):
                st.session_state.override = False
                add_event(st.session_state.facility, "Блокировка сценария снята", "Информация", "—", "Пользователь вернул демонстрационный сценарий в исходный режим управления.")
                st.rerun()
        else:
            st.markdown(
                '<div class="glass-card"><h4>Ожидание события</h4><p>Система отображает текущий условный профиль и ожидает новых точек данных или запуска аварийного сценария.</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Архитектура обработки данных</div>', unsafe_allow_html=True)
    architecture_columns = st.columns(4, gap="medium")
    architecture = [
        ("01 · IoT", "Датчики тока, напряжения, температуры, изоляции и баланса токов."),
        ("02 · Pipeline", "Передача телеметрии от устройств к серверной части."),
        ("03 · AI Core", "Анализ параметров, обнаружение аномалий и прогнозирование."),
        ("04 · UI", "Графики, статусы, журнал событий и рекомендации диспетчеру."),
    ]
    for col, (heading, description) in zip(architecture_columns, architecture):
        with col:
            st.markdown(
                f'<div class="glass-card"><h4>{html.escape(heading)}</h4><p>{html.escape(description)}</p></div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# 11. ВКЛАДКА «СИМУЛЯТОР»
# =============================================================================

if main_page == "Симулятор":
    st.markdown('<div class="section-heading">Проверка параметров через AI Core</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-card"><p>Измените значения тока, напряжения и температуры. Модель сравнит выбранную комбинацию с синтетическим профилем и покажет демонстрационную оценку. Пороговые значения не являются уставками промышленной защиты.</p></div>',
        unsafe_allow_html=True,
    )

    sim_columns = st.columns(3, gap="large")
    with sim_columns[0]:
        simulator_current = st.slider("Ток нагрузки, А", 0, 100, 15, 1, key="sim_current")
    with sim_columns[1]:
        simulator_voltage = st.slider("Напряжение сети, В", 300, 450, 380, 1, key="sim_voltage")
    with sim_columns[2]:
        simulator_temperature = st.slider("Температура оборудования, °C", 0, 150, 60, 1, key="sim_temperature")

    check_col, check_info = st.columns([1, 2], vertical_alignment="center")
    with check_col:
        run_check = st.button("Проверить параметры через ИИ", key="run_ai_check", width="stretch")
    with check_info:
        st.caption(
            f"Проверяемая комбинация: I = {simulator_current} А · U = {simulator_voltage} В · T = {simulator_temperature} °C"
        )

    if run_check:
        model = get_ai_model()
        sample = pd.DataFrame(
            [{"Ток": float(simulator_current), "Напряжение": float(simulator_voltage), "Температура": float(simulator_temperature)}]
        )
        prediction = int(model.predict(sample)[0])
        score = float(model.decision_function(sample)[0])
        critical = (
            simulator_current > CURRENT_CRITICAL_A
            or simulator_temperature >= TEMPERATURE_CRITICAL_C
            or simulator_voltage < 355
            or simulator_voltage > 405
        )
        warning = (
            simulator_current > CURRENT_WARNING_A
            or simulator_temperature > TEMPERATURE_WARNING_C
            or simulator_voltage < VOLTAGE_LOW_V
            or simulator_voltage > VOLTAGE_HIGH_V
        )
        st.session_state.simulator_result = {
            "critical": critical,
            "warning": warning,
            "prediction": prediction,
            "score": score,
            "current": simulator_current,
            "voltage": simulator_voltage,
            "temperature": simulator_temperature,
        }

    if st.session_state.simulator_result is not None:
        result = st.session_state.simulator_result
        if result["critical"]:
            st.markdown(
                "<div class='emergency-alert'><strong>КРИТИЧЕСКОЕ ОТКЛОНЕНИЕ</strong><p>Один или несколько параметров вышли за демонстрационные критические пороги.</p></div>",
                unsafe_allow_html=True,
            )
        elif result["warning"] or result["prediction"] == -1:
            st.warning("Обнаружено отклонение от условного рабочего профиля. Рекомендуется проверить исходные данные и состояние оборудования.")
        else:
            st.success("В пределах демонстрационного профиля. Статистическая модель не обнаружила выраженную аномалию.")

        result_cols = st.columns(4)
        result_cols[0].metric("Ток", f"{result['current']} А")
        result_cols[1].metric("Напряжение", f"{result['voltage']} В")
        result_cols[2].metric("Температура", f"{result['temperature']} °C")
        result_cols[3].metric("IsolationForest", "Аномалия" if result["prediction"] == -1 else "Норма", f"score {result['score']:.4f}")
    else:
        st.info("Задайте параметры и нажмите «Проверить параметры через ИИ».")

    st.markdown(
        """
        <div class="glass-card">
            <h4>Как интерпретируется результат</h4>
            <p>Температура 50–70 °C приведена в статье как ориентировочный нормальный диапазон для трансформаторного оборудования; нагрев до 90–100 °C и выше рассматривается как возможный признак опасного состояния. Значение тока 10–15 А в статье приведено как пример штатного диапазона, а рост до 50 А — как пример перегрузки или короткого замыкания.</p>
            <p>В симуляторе дополнительно используются условные пороги напряжения и тока, которые нужны для демонстрации интерфейса. Для реального оборудования диапазоны должны устанавливаться по проектной и эксплуатационной документации.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 12. ВКЛАДКА «АРХИВ И ОБСЛУЖИВАНИЕ»
# =============================================================================

if main_page == "Архив и обслуживание":
    st.markdown('<div class="section-heading">Архив измерений и техническая история</div>', unsafe_allow_html=True)
    st.caption("Оперативный график мониторинга ограничен последними 365 днями. Более старые данные, когда они накопятся, переносятся в отдельную архивную таблицу и доступны здесь.")

    archive_section = st.radio(
        "Разделы архива",
        ["История измерений", "История ремонтов", "Замена оборудования", "Журнал событий"],
        horizontal=True,
        key="archive_section",
        label_visibility="collapsed",
    )

    if archive_section == "История измерений":
        filters = st.columns(4, gap="medium")
        with filters[0]:
            archive_facility = st.selectbox("Объект", FACILITIES, key="archive_facility")
        with filters[1]:
            archive_parameter = st.selectbox("Параметр", list(PARAMETERS.keys()), key="archive_parameter")
        with filters[2]:
            archive_start = st.date_input(
                "Дата начала",
                value=date.today() - timedelta(days=30),
                min_value=date.today() - timedelta(days=365 * 5),
                max_value=date.today(),
                key="archive_start_date",
            )
        with filters[3]:
            archive_end = st.date_input(
                "Дата окончания",
                value=date.today(),
                min_value=date.today() - timedelta(days=365 * 5),
                max_value=date.today(),
                key="archive_end_date",
            )

        if archive_start > archive_end:
            st.warning("Дата начала не может быть позже даты окончания.")
            archive_data = pd.DataFrame(columns=["Время", "Ток", "Напряжение", "Температура"])
        else:
            archive_data = read_telemetry(
                archive_facility,
                datetime.combine(archive_start, datetime.min.time()),
                datetime.combine(archive_end, datetime.max.time().replace(microsecond=0)),
                include_long_archive=True,
            )

        if archive_data.empty:
            st.info("За выбранный период записи не найдены. Для демонстрации доступны синтетические данные примерно за последний год.")
        else:
            archive_fig = make_history_chart(
                archive_data,
                archive_parameter,
                f"{archive_facility} · {archive_start:%d.%m.%Y} — {archive_end:%d.%m.%Y}",
            )
            st.plotly_chart(
                archive_fig,
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
                key="long_term_archive_chart",
            )
            summary_cols = st.columns(4)
            values = archive_data[archive_parameter]
            meta = PARAMETERS[archive_parameter]
            summary_cols[0].metric("Количество точек", f"{len(values):,}".replace(",", " "))
            summary_cols[1].metric("Минимум", f"{values.min():.2f} {meta['unit']}")
            summary_cols[2].metric("Среднее", f"{values.mean():.2f} {meta['unit']}")
            summary_cols[3].metric("Максимум", f"{values.max():.2f} {meta['unit']}")
            export_frame = archive_data[["Время", archive_parameter]].copy()
            export_frame["Время"] = export_frame["Время"].dt.strftime("%Y-%m-%d %H:%M:%S")
            export_frame = export_frame.rename(columns={archive_parameter: f"{archive_parameter}, {meta['unit']}"})
            st.download_button(
                "Скачать выбранный период (CSV)",
                data=export_frame.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"archive_{archive_parameter.lower()}_{archive_start}_{archive_end}.csv",
                mime="text/csv",
                key="download_archive_csv",
            )

    if archive_section == "История ремонтов":
        st.markdown('<div class="section-heading">Журнал ремонтных работ</div>', unsafe_allow_html=True)
        repairs = read_repairs()
        if repairs.empty:
            st.info("Записей о ремонте пока нет. Ниже можно добавить первую запись. Все записи сохраняются в локальной базе данных приложения.")
        else:
            st.dataframe(repairs, hide_index=True, width="stretch", height=260)

        with st.expander("Добавить запись о ремонте", expanded=repairs.empty):
            with st.form("add_repair_form", clear_on_submit=True):
                repair_cols = st.columns(2)
                repair_facility = repair_cols[0].selectbox("Объект", FACILITIES, key="repair_facility_input")
                repair_date = repair_cols[1].date_input("Дата ремонта", value=date.today(), key="repair_date_input")
                equipment_name = st.text_input("Наименование оборудования", placeholder="Например, контактное соединение К-01")
                reason = st.text_input("Причина ремонта", placeholder="Причина или выявленный дефект")
                work_done = st.text_area("Выполненные работы", placeholder="Кратко опишите выполненные работы")
                engineer = st.text_input("Ответственный инженер", placeholder="ФИО или табельный идентификатор")
                result = st.text_input("Результат проверки", placeholder="Например, контрольные параметры в норме")
                save_repair = st.form_submit_button("Сохранить запись о ремонте", width="stretch")
                if save_repair:
                    if not all([equipment_name.strip(), reason.strip(), work_done.strip(), engineer.strip(), result.strip()]):
                        st.warning("Заполните все поля записи о ремонте.")
                    else:
                        insert_repair(repair_facility, repair_date, equipment_name.strip(), reason.strip(), work_done.strip(), engineer.strip(), result.strip())
                        add_event(repair_facility, "Добавлена запись о ремонте", "Информация", "—", f"Оборудование: {equipment_name.strip()}. Ответственный: {engineer.strip()}.")
                        st.success("Запись о ремонте сохранена.")
                        st.rerun()

    if archive_section == "Замена оборудования":
        st.markdown('<div class="section-heading">Реестр оборудования и планирование замены</div>', unsafe_allow_html=True)
        equipment_data = read_equipment()
        if equipment_data.empty:
            st.info("Реестр пока пуст. Добавьте оборудование по данным его паспорта: производителя, модель, заводской номер, дату установки, срок службы и контакты производителя.")
        else:
            display_equipment = equipment_data.copy()
            display_equipment["Плановая замена"] = display_equipment.apply(
                lambda row: (
                    (pd.to_datetime(row["Дата установки"]) + pd.DateOffset(years=int(row["Срок эксплуатации, лет"]))).date().isoformat()
                    if row["Дата установки"] and int(row["Срок эксплуатации, лет"]) > 0
                    else "Не рассчитана"
                ),
                axis=1,
            )
            st.dataframe(display_equipment, hide_index=True, width="stretch", height=270)

        with st.expander("Добавить оборудование в реестр", expanded=equipment_data.empty):
            with st.form("add_equipment_form", clear_on_submit=True):
                equipment_facility = st.selectbox("Объект установки", FACILITIES, key="equipment_facility_input")
                equipment_name_input = st.text_input("Наименование оборудования", placeholder="Например, силовой трансформатор")
                equip_cols = st.columns(2)
                manufacturer = equip_cols[0].text_input("Производитель", placeholder="Название предприятия")
                model = equip_cols[1].text_input("Модель", placeholder="Марка / модель")
                serial_number = st.text_input("Заводской номер", placeholder="Номер с заводской таблички")
                date_and_life = st.columns(2)
                install_date = date_and_life[0].date_input("Дата установки / замены", value=date.today(), key="equipment_install_date")
                service_life_years = date_and_life[1].number_input("Срок эксплуатации, лет", min_value=1, max_value=100, value=25, step=1)
                manufacturer_contact = st.text_input("Контакты производителя", placeholder="Телефон, e-mail или официальный сайт")
                equipment_notes = st.text_area("Примечание", placeholder="Паспортные данные, гарантия, особенности обслуживания")
                save_equipment = st.form_submit_button("Сохранить оборудование", width="stretch")
                if save_equipment:
                    if not all([equipment_name_input.strip(), manufacturer.strip(), model.strip(), serial_number.strip(), manufacturer_contact.strip()]):
                        st.warning("Заполните наименование, производителя, модель, заводской номер и контакты производителя.")
                    else:
                        insert_equipment(
                            equipment_facility,
                            equipment_name_input.strip(),
                            manufacturer.strip(),
                            model.strip(),
                            serial_number.strip(),
                            install_date,
                            int(service_life_years),
                            manufacturer_contact.strip(),
                            equipment_notes.strip(),
                        )
                        add_event(equipment_facility, "Добавлено оборудование", "Информация", "—", f"В реестр добавлено оборудование: {equipment_name_input.strip()}.")
                        st.success("Оборудование добавлено в реестр.")
                        st.rerun()

    if archive_section == "Журнал событий":
        st.markdown('<div class="section-heading">Журнал событий системы</div>', unsafe_allow_html=True)
        event_facility = st.selectbox("Фильтр по объекту", ["Все объекты"] + FACILITIES, key="events_facility_filter")
        event_data = read_events(None if event_facility == "Все объекты" else event_facility, limit=1000)
        if event_data.empty:
            st.info("Событий пока нет.")
        else:
            st.dataframe(event_data, hide_index=True, width="stretch", height=420)
            st.download_button(
                "Скачать журнал событий (CSV)",
                data=event_data.to_csv(index=False).encode("utf-8-sig"),
                file_name="substation_events.csv",
                mime="text/csv",
                key="download_events_csv",
            )


# =============================================================================
# 13. ВКЛАДКА «ДОКУМЕНТАЦИЯ»
# =============================================================================

if main_page == "Документация":
    st.markdown('<div class="section-heading">Документация проектируемой системы</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Справочные материалы объединяют требования, роли пользователей, основные понятия и правила безопасной работы с демонстрационной платформой.</p>',
        unsafe_allow_html=True,
    )

    docs_col1, docs_col2, docs_col3 = st.columns([1.35, 1.0, 1.0], gap="large")

    with docs_col1:
        with st.container(key="docs-requirements-card"):
            st.markdown(
                '<div class="doc-card-symbol">▤</div><div class="doc-card-title">Требования системы</div><div class="doc-card-desc">Функциональные и нефункциональные требования, сформированные в проектной статье.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Функциональные требования  ·  F-01—F-12", key="open_functional_dialog", width="stretch"):
                show_functional_requirements()
            if st.button("Нефункциональные требования  ·  NF-01—NF-05", key="open_nonfunctional_dialog", width="stretch"):
                show_nonfunctional_requirements()

    with docs_col2:
        with st.container(key="docs-roles-card"):
            st.markdown(
                '<div class="doc-card-symbol">♙</div><div class="doc-card-title">Роли и права доступа</div><div class="doc-card-desc">Обязанности диспетчера и инженера, доступные функции и ограничения.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Открыть роли и обязанности", key="open_roles_dialog", width="stretch"):
                show_roles_dialog()

    with docs_col3:
        with st.container(key="docs-glossary-card"):
            st.markdown(
                '<div class="doc-card-symbol">≡</div><div class="doc-card-title">Термины и правила</div><div class="doc-card-desc">Электротехнические определения, архитектура ИИ-платформы и правила использования.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Открыть справочник", key="open_glossary_dialog", width="stretch"):
                show_glossary_dialog()

    st.markdown('<div class="section-heading" style="margin-top:1.55rem;">Что реализовано в текущем прототипе</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <p>В интерфейсе представлены демонстрационные измерения тока, напряжения и температуры; графики с ориентировочными диапазонами и средними значениями; проверка аномалий с помощью IsolationForest; аварийный сценарий; архив измерений; журнал событий; реестр оборудования и история ремонтных работ.</p>
            <p>Историческая телеметрия хранится в локальной базе SQLite. Оперативный раздел использует данные за последние 365 дней, а устаревающие точки переносятся в долговременную архивную таблицу. Сведения о ремонтах и оборудовании сохраняются локально между перезапусками приложения.</p>
            <p>Подключение реальных датчиков IoT, промышленного брокера сообщений, защищенной авторизации, устройств релейной защиты и исполнительных механизмов потребует самостоятельной разработки, интеграции и испытаний.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 14. ПОДВАЛ
# =============================================================================

st.markdown("---")
st.markdown(
    f'<div style="text-align:center;color:#64748B;font-size:0.72rem;letter-spacing:0.12em;">ПОДСТАНЦИЯ АПК · DEMO BUILD · {html.escape(AUTHOR_FULL_NAME)}</div>',
    unsafe_allow_html=True,
)
