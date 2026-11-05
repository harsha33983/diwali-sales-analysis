"""
Metrics and KPI Presentation Component.
Renders responsive, premium Diwali-themed KPI cards.
"""

import streamlit as st


def apply_custom_css():
    """Inject custom CSS for dark navy theme, golden-orange accents, and rounded card styling."""
    custom_css = """
    <style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Branded Header Banner */
    .diwali-header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(230, 126, 34, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .diwali-title {
        color: #F8FAFC;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .diwali-title span {
        color: #E67E22;
        background: -webkit-linear-gradient(45deg, #F39C12, #E67E22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .diwali-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #E67E22;
        box-shadow: 0 6px 16px rgba(230, 126, 34, 0.15);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #E67E22, #F39C12);
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }

    .kpi-subtext {
        font-size: 11px;
        color: #CBD5E1;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .kpi-subtext .highlight {
        color: #F39C12;
        font-weight: 600;
    }

    /* Section Containers */
    .dashboard-section {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Card */
    .prediction-box {
        background: linear-gradient(135deg, #1E293B 0%, #172554 100%);
        border: 1px solid #F59E0B;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 38px;
        font-weight: 800;
        color: #FBBF24;
        margin: 12px 0;
    }
    /* Hero Metric Box */
    .hero-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .hero-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #F59E0B;
        margin-bottom: 8px;
    }

    .hero-value-row {
        display: flex;
        align-items: baseline;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }

    .hero-value {
        font-size: 42px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }

    .hero-raw {
        font-size: 18px;
        font-weight: 600;
        color: #94A3B8;
    }

    .hero-raw span {
        color: #FBBF24;
    }

    .hero-subtext {
        font-size: 14px;
        color: #CBD5E1;
        margin-bottom: 16px;
    }

    .hero-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    .hero-pill {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        color: #E2E8F0;
        font-weight: 500;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge: str = "Diwali Retail Analytics"):
    """Render consistent branded top bar across dashboard pages."""
    apply_custom_css()
    html = (
        f'<div class="diwali-header-box">'
        f'<div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #F59E0B; font-weight: 700; margin-bottom: 4px;">🪔 {badge}</div>'
        f'<h1 class="diwali-title">{title}</h1>'
        f'<div class="diwali-subtitle">{subtitle}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_hero_box(
    title: str = "GROSS FESTIVAL TURNOVER",
    value: str = "₹10.62 Cr",
    total_raw: str = "₹106,178,828",
    subtitle: str = "11,231 Verified Transactions • 16 Indian States",
    highlights: list[str] = None
):
    """Render a dedicated, prominent Hero Metric Box."""
    apply_custom_css()
    pills_html = ""
    if highlights:
        pills_markup = "".join([f'<div class="hero-pill">{h}</div>' for h in highlights])
        pills_html = f'<div class="hero-pills">{pills_markup}</div>'

    html = (
        f'<div class="hero-box">'
        f'<div class="hero-label">🪔 {title}</div>'
        f'<div class="hero-value-row">'
        f'<div class="hero-value">{value}</div>'
        f'<div class="hero-raw">Total Gross: <span>{total_raw}</span></div>'
        f'</div>'
        f'<div class="hero-subtext">{subtitle}</div>'
        f'{pills_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", highlight: str = "") -> str:
    """Render a single responsive HTML KPI card with zero indentation issues."""
    sub_markup = ""
    if subtext or highlight:
        sub_markup = f'<div class="kpi-subtext"><span>{subtext}</span> <span class="highlight">{highlight}</span></div>'

    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_markup}'
        f'</div>'
    )


def render_kpi_grid(kpis: list[dict]):
    """Render a responsive flexbox/grid of KPI cards without markdown indentation issues."""
    cards_html = "".join([
        render_kpi_card(
            label=k["label"],
            value=k["value"],
            subtext=k.get("subtext", ""),
            highlight=k.get("highlight", "")
        )
        for k in kpis
    ])
    html = f'<div class="kpi-container">{cards_html}</div>'
    st.markdown(html, unsafe_allow_html=True)

