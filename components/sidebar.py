"""
Global Sidebar Component.
Provides multi-dimensional slice-and-dice interactive filters across dashboard pages.
"""

import streamlit as st
import pandas as pd


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renders filter controls in Streamlit sidebar and returns filtered DataFrame.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 24px;">🪔</span>
                    <div>
                        <div style="font-weight: 700; color: #F8FAFC; font-size: 16px;">Diwali Sales BI</div>
                        <div style="font-size: 11px; color: #F59E0B;">Enterprise Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Interactive Filters")

        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state["filter_state"] = []
            st.session_state["filter_gender"] = []
            st.session_state["filter_age"] = []
            st.session_state["filter_occ"] = []
            st.session_state["filter_cat"] = []
            st.rerun()

        # States Filter
        all_states = sorted(df["State"].dropna().unique().tolist())
        selected_states = st.multiselect(
            "State / Region",
            options=all_states,
            default=st.session_state.get("filter_state", []),
            placeholder="All States Selected",
            key="sb_states"
        )

        # Gender Filter
        all_genders = sorted(df["Gender_Desc"].dropna().unique().tolist())
        selected_genders = st.multiselect(
            "Gender",
            options=all_genders,
            default=st.session_state.get("filter_gender", []),
            placeholder="All Genders Selected",
            key="sb_genders"
        )

        # Age Group Filter
        all_age_groups = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
        present_age_groups = [ag for ag in all_age_groups if ag in df["Age Group"].unique()]
        selected_age_groups = st.multiselect(
            "Age Bracket",
            options=present_age_groups,
            default=st.session_state.get("filter_age", []),
            placeholder="All Age Brackets Selected",
            key="sb_ages"
        )

        # Product Category Filter
        all_cats = sorted(df["Product_Category"].dropna().unique().tolist())
        selected_cats = st.multiselect(
            "Product Category",
            options=all_cats,
            default=st.session_state.get("filter_cat", []),
            placeholder="All Categories Selected",
            key="sb_cats"
        )

        # Occupation Filter
        all_occs = sorted(df["Occupation"].dropna().unique().tolist())
        selected_occs = st.multiselect(
            "Customer Occupation",
            options=all_occs,
            default=st.session_state.get("filter_occ", []),
            placeholder="All Occupations Selected",
            key="sb_occs"
        )

        # Apply filtering
        filtered_df = df.copy()
        if selected_states:
            filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
        if selected_genders:
            filtered_df = filtered_df[filtered_df["Gender_Desc"].isin(selected_genders)]
        if selected_age_groups:
            filtered_df = filtered_df[filtered_df["Age Group"].isin(selected_age_groups)]
        if selected_cats:
            filtered_df = filtered_df[filtered_df["Product_Category"].isin(selected_cats)]
        if selected_occs:
            filtered_df = filtered_df[filtered_df["Occupation"].isin(selected_occs)]

        # Dataset Status Indicator
        total_rows = len(df)
        filtered_rows = len(filtered_df)
        pct = (filtered_rows / total_rows) * 100

        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 20px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Active Cohort Slice</div>
                <div style="font-size: 18px; font-weight: 700; color: #38BDF8; margin-top: 4px;">{filtered_rows:,} / {total_rows:,}</div>
                <div style="font-size: 11px; color: #64748B;">{pct:.1f}% of entire Diwali dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="font-size: 11px; color: #64748B; margin-top: 24px; text-align: center;">
                Diwali Analytics Platform v2.0<br/>Powered by Streamlit & Scikit-Learn
            </div>
            """,
            unsafe_allow_html=True
        )

    return filtered_df

