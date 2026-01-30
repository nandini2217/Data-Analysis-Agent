import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

from agents.data_profiler import profile_data
from agents.chat_agent import answer_question
from utils.document_reader import extract_text

# ---------- GLOBAL CHART SETTINGS ----------
plt.rcParams.update({
    "figure.figsize": (2.2, 1.8),   # SMALL charts
    "figure.dpi": 100,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6
})

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Data Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

# ---------- FILE UPLOAD ----------
csv_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
doc_files = st.file_uploader(
    "📄 Upload Supporting Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

df = None
document_text = ""

if csv_file:
    df = pd.read_csv(csv_file)
    st.success("✅ CSV uploaded successfully")

if doc_files:
    document_text = extract_text(doc_files)

# ==================================================
# ================= DASHBOARD ======================
# ==================================================
if df is not None:
    st.markdown("---")
    st.subheader("📊 Dashboard Controls")

    c1, c2, c3 = st.columns(3)

    with c1:
        dashboard_title = st.text_input(
            "Dashboard Title",
            "One Page Business Dashboard",
            key="title"
        )

    with c2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Pie", "Histogram"],
            key="chart_type"
        )

    with c3:
        theme = st.selectbox(
            "Theme",
            ["default", "ggplot", "seaborn"],
            key="theme"
        )

    color = st.color_picker("Chart Color", "#4CAF50")

    selected_cols = st.multiselect(
        "Select up to 4 columns",
        df.columns.tolist(),
        key="cols"
    )[:4]

    summary = st.text_area(
        "Dashboard Summary",
        "This dashboard summarizes key insights from the dataset.",
        key="summary"
    )

    generate = st.button("🚀 Generate One-Page Dashboard")

    # ================= RENDER DASHBOARD =================
    if generate and selected_cols:

        if theme != "default":
            plt.style.use(theme)

        st.markdown("---")
        st.subheader(dashboard_title)
        st.caption(summary)

        # ---------- KPIs ----------
        k1, k2, k3 = st.columns(3)
        k1.metric("Rows", df.shape[0])
        k2.metric("Columns", df.shape[1])
        k3.metric(
            "Missing %",
            f"{round(df.isna().mean().mean() * 100, 2)}%"
        )

        # ---------- CHART GRID ----------
        st.markdown("### 📈 Visuals (One Page)")
        chart_grid = st.columns(2)

        chart_images = []

        for i, col in enumerate(selected_cols):
            with chart_grid[i % 2]:
                fig, ax = plt.subplots(figsize=(2.4, 2.0))

                if chart_type == "Histogram" and pd.api.types.is_numeric_dtype(df[col]):
                    ax.hist(df[col].dropna(), bins=15, color=color, edgecolor='black')
                    ax.set_xlabel("Value Range", fontsize=7)
                    ax.set_ylabel("Frequency", fontsize=7)

                elif chart_type == "Bar":
                    value_counts = df[col].value_counts().head(6)
                    value_counts.plot(kind="bar", ax=ax, color=color)
                    ax.set_xlabel(col, fontsize=7)
                    ax.set_ylabel("Count", fontsize=7)
                    ax.tick_params(axis='x', rotation=45)

                elif chart_type == "Pie":
                    counts = df[col].value_counts().head(5)
                    ax.pie(
                        counts,
                        labels=counts.index,
                        autopct="%1.0f%%",
                        startangle=90,
                        radius=0.55,
                        textprops={'fontsize': 7},
                    )
                    

                elif chart_type == "Line" and pd.api.types.is_numeric_dtype(df[col]):
                    ax.plot(df[col].reset_index(drop=True), linewidth=1, color=color)
                    ax.set_xlabel("Index", fontsize=7)
                    ax.set_ylabel(col, fontsize=7)

                ax.set_title(col, fontsize=9, pad=1)
                plt.tight_layout(pad=0.2)
                st.pyplot(fig, use_container_width=False)

                # Save chart for PDF
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp.name, bbox_inches="tight")
                chart_images.append(tmp.name)

        # ==================================================
        # ================= PDF EXPORT =====================
        # ==================================================
        def create_pdf():
            pdf_path = "one_page_dashboard.pdf"
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=landscape(A4)
            )

            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph(dashboard_title, styles["Title"]))
            elements.append(Paragraph(summary, styles["Normal"]))

            kpi_table = Table([
                ["Rows", "Columns", "Missing %"],
                [
                    df.shape[0],
                    df.shape[1],
                    f"{round(df.isna().mean().mean() * 100, 2)}%"
                ]
            ])
            elements.append(kpi_table)

            for img in chart_images:
                elements.append(Image(img, width=3.0, height=2.4))

            doc.build(elements)
            return pdf_path

        pdf_file = create_pdf()

        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇ Download Dashboard (PDF)",
                f,
                "one_page_dashboard.pdf",
                "application/pdf"
            )

# ==================================================
# ================= CHAT AGENT =====================
# ==================================================
st.markdown("---")
st.subheader("💬 Ask Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask anything about your data")

if st.button("Ask") and query and df is not None:
    answer = answer_question(query, df, document_text)
    st.session_state.chat_history.append((query, answer))

for q, a in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {q}")
    st.success(a if isinstance(a, str) else str(a))
