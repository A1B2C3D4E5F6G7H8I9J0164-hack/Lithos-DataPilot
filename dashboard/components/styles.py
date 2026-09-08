"""
Design system, CSS styling, and reusable HTML component primitives for Autonomous Data Scientist.
Infused with the high-end Lithos dark luxury aesthetic: Playfair Display typography, warm copper/amber
accents (#E8702A), obsidian surfaces, pill geometries, and glassmorphic micro-interactions.
"""
import streamlit as st


def inject_custom_css():
    """Injects high-end dark-first design system styles into Streamlit DOM."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,500;1,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0B0E;
        --bg-surface: #141419;
        --bg-surface-elevated: #1C1D24;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(232, 112, 42, 0.35);
        --border-hover: rgba(232, 112, 42, 0.5);
        --text-primary: #F8FAFC;
        --text-secondary: #9E9EA8;
        --text-muted: #666675;
        --accent-orange: #E8702A;
        --accent-amber: #F59E0B;
        --accent-copper: #D4601C;
        --accent-emerald: #10B981;
        --accent-blue: #38BDF8;
        --accent-rose: #F43F5E;
    }

    /* Core typography and background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Editorial serif typography for headings & accents */
    h1, h2, h3, .font-playfair {
        font-family: 'Playfair Display', Georgia, serif !important;
        letter-spacing: -0.02em !important;
    }

    .font-playfair em, .font-playfair i, .ads-italic {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-style: italic !important;
        font-weight: 400 !important;
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px !important;
    }

    /* Custom Lithos-styled Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0B0E;
    }
    ::-webkit-scrollbar-thumb {
        background: #252630;
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #E8702A;
    }

    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #08080A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        padding-top: 1.5rem !important;
    }

    /* Sidebar Radio Buttons - Pill shaped */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent !important;
        border-radius: 9999px !important;
        padding: 0.55rem 1rem !important;
        margin-bottom: 0.25rem !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(232, 112, 42, 0.08) !important;
        border-color: rgba(232, 112, 42, 0.2) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(232, 112, 42, 0.15) !important;
        border: 1px solid rgba(232, 112, 42, 0.45) !important;
        color: #FFFFFF !important;
    }

    /* Cards & Glassmorphic Containers */
    .ads-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .ads-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-1px);
    }
    .ads-card-elevated {
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-accent);
    }

    /* Pill Badges */
    .ads-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .ads-badge-orange, .ads-badge-blue {
        background: rgba(232, 112, 42, 0.12);
        color: #F97316;
        border: 1px solid rgba(232, 112, 42, 0.32);
    }
    .ads-badge-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .ads-badge-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }
    .ads-badge-rose {
        background: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.28);
    }

    /* Hero Banner with Spotlight Radial Glow */
    .ads-hero {
        position: relative;
        overflow: hidden;
        background: radial-gradient(ellipse at 50% -20%, rgba(232, 112, 42, 0.25) 0%, rgba(20, 20, 25, 0.95) 60%, #0B0B0E 100%);
        border: 1px solid rgba(232, 112, 42, 0.3);
        border-radius: 16px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .ads-hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
        line-height: 1.15;
    }
    .ads-hero-title em, .ads-hero-title i {
        font-style: italic;
        font-weight: 400;
        color: #F97316;
    }
    .ads-hero-subtitle {
        font-size: 1.05rem;
        color: #9E9EA8;
        line-height: 1.6;
        max-width: 720px;
        margin-bottom: 1.5rem;
    }

    /* Pipeline Flow Step Box */
    .pipeline-step-item {
        display: flex;
        align-items: center;
        padding: 0.85rem 1.1rem;
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
    }
    .pipeline-step-item.active {
        border-color: var(--accent-orange);
        background: rgba(232, 112, 42, 0.09);
        box-shadow: 0 0 15px rgba(232, 112, 42, 0.15);
    }
    .pipeline-step-item.completed {
        border-color: rgba(16, 185, 129, 0.45);
    }

    /* Terminal Console */
    .ads-terminal {
        background-color: #060608;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #D1D5DB;
        max-height: 290px;
        overflow-y: auto;
        line-height: 1.65;
    }
    .ads-log-timestamp {
        color: #666675;
        margin-right: 0.5rem;
    }
    .ads-log-decision {
        color: #F97316;
        font-weight: 600;
    }

    /* Pill Buttons & Interactive Elements */
    div.stButton > button {
        background: #181920 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9999px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button:hover {
        background: #23242E !important;
        border-color: var(--accent-orange) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(232, 112, 42, 0.25) !important;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: #E8702A !important;
        border: 1px solid #F97316 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(232, 112, 42, 0.45) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #D4601C !important;
        box-shadow: 0 6px 25px rgba(232, 112, 42, 0.6) !important;
    }

    /* Metric Stat Box */
    .ads-stat-box {
        background: #141419;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        transition: border-color 0.2s ease;
    }
    .ads-stat-box:hover {
        border-color: rgba(232, 112, 42, 0.3);
    }
    .ads-stat-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #9E9EA8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .ads-stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .ads-stat-delta {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Custom Streamlit Input Styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #141419 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #E8702A !important;
        box-shadow: 0 0 0 1px #E8702A !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "", help_text: str = ""):
    """Renders a sleek statistic card with Lithos styling."""
    delta_html = f'<div class="ads-stat-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="ads-stat-box">
        <div class="ads-stat-label" title="{help_text}">{label}</div>
        <div class="ads-stat-value">{value}</div>
        {delta_html}
    </div>
    """


def render_badge(text: str, badge_type: str = "orange") -> str:
    """Renders a styled chip/badge."""
    return f'<span class="ads-badge ads-badge-{badge_type}">{text}</span>'
