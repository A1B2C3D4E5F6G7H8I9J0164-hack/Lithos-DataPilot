"""
Design system, CSS styling, and reusable HTML component primitives for Autonomous Data Scientist.
"""
import streamlit as st


def inject_custom_css():
    """Injects high-end dark-first design system styles into Streamlit DOM."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0F17;
        --bg-surface: #121826;
        --bg-surface-elevated: #1A2336;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(59, 130, 246, 0.3);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #3B82F6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #EF4444;
    }

    /* Core typography and background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #080C14 !important;
        border-right: 1px solid var(--border-color) !important;
        padding-top: 1.5rem !important;
    }

    /* Cards & Containers */
    .ads-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease;
    }
    .ads-card:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }
    .ads-card-elevated {
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-accent);
    }

    /* Metric Badges */
    .ads-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .ads-badge-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .ads-badge-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .ads-badge-rose {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .ads-badge-blue {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Hero Banner */
    .ads-hero {
        background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.15) 0%, transparent 60%),
                    linear-gradient(180deg, #121A2B 0%, #0B0F17 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .ads-hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .ads-hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.5;
        max-width: 650px;
        margin-bottom: 1.5rem;
    }

    /* Pipeline Flow Step Box */
    .pipeline-step-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .pipeline-step-item.active {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.08);
    }
    .pipeline-step-item.completed {
        border-color: rgba(16, 185, 129, 0.4);
    }

    /* Terminal Console */
    .ads-terminal {
        background-color: #06090F;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #CBD5E1;
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.6;
    }
    .ads-log-timestamp {
        color: #64748B;
        margin-right: 0.5rem;
    }
    .ads-log-decision {
        color: #38BDF8;
        font-weight: 600;
    }

    /* Interactive buttons */
    div.stButton > button {
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #2D3B52;
        border-color: var(--accent-blue);
        color: #FFFFFF;
    }
    div.stButton > button[kind="primary"] {
        background: #2563EB;
        border-color: #3B82F6;
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #1D4ED8;
    }

    /* Metric cards */
    .ads-stat-box {
        background: #101624;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .ads-stat-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .ads-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .ads-stat-delta {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "", help_text: str = ""):
    """Renders a sleek statistic card."""
    delta_html = f'<div class="ads-stat-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="ads-stat-box">
        <div class="ads-stat-label" title="{help_text}">{label}</div>
        <div class="ads-stat-value">{value}</div>
        {delta_html}
    </div>
    """


def render_badge(text: str, badge_type: str = "blue") -> str:
    """Renders a styled chip/badge."""
    return f'<span class="ads-badge ads-badge-{badge_type}">{text}</span>'
