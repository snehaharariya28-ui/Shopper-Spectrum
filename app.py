# Shopper Spectrum — Customer Segmentation & Product Recommendation

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — only layout tweaks, no color overrides ──────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8F7F4; }
[data-testid="stSidebar"] { background: #1C1C1E; }
[data-testid="stSidebar"] * { color: #F5F5F0 !important; }
[data-testid="stSidebar"] hr { border-color: #3F3F46; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #F5F5F0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label span { color: #F5F5F0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #71717A !important; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label { font-size: 13px !important; color: #71717A !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] { color: #0D9488 !important; font-weight: 800 !important; }
.stButton > button {
    background: #0D9488 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    transition: background 0.15s;
}
.stButton > button:hover { background: #0F766E !important; }
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 8px;
}
.seg-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    border-left: 5px solid;
}
.insight-row {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #1C1C1E;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly base template ───────────────────────────────────────────────────────
TEAL       = "#0D9488"
DARK_TEAL  = "#0F766E"
TEAL_LIGHT = "#5EEAD4"
AMBER      = "#F59E0B"
CORAL      = "#E11D48"
PURPLE     = "#7C3AED"
SLATE      = "#64748B"
BG         = "#F8F7F4"
CARD       = "#FFFFFF"

SEG_COLORS = {
    "High-Value": "#0D9488",
    "Regular":    "#3B82F6",
    "Occasional": "#F59E0B",
    "At-Risk":    "#E11D48",
}
SEG_ICONS = {
    "High-Value": "💎",
    "Regular":    "📦",
    "Occasional": "🔄",
    "At-Risk":    "⚠️",
}

AXIS = dict(
    gridcolor="#EBEBEB",
    linecolor="#D4D4D4",
    tickcolor="#52525B",
    tickfont=dict(color="#52525B", size=11),
    title_font=dict(color="#52525B", size=12),
    zerolinecolor="#D4D4D4",
)

def fig_layout(fig, height=400, legend=True):
    fig.update_layout(
        height=height,
        plot_bgcolor=CARD,
        paper_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color="#52525B", size=12),
        title_font=dict(color="#27272A", size=14, family="Inter, sans-serif"),
        margin=dict(l=16, r=16, t=40, b=16),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color="#52525B"),
        ),
        xaxis=AXIS,
        yaxis=AXIS,
    )
    return fig

# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_data.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    elbow = pd.read_csv("data/elbow_data.csv")
    return df, elbow

@st.cache_resource
def load_models():
    km    = pickle.load(open("models/kmeans_model.pkl",      "rb"))
    sc    = pickle.load(open("models/scaler.pkl",            "rb"))
    c2s   = pickle.load(open("models/cluster_to_segment.pkl","rb"))
    rfm   = pickle.load(open("models/rfm_data.pkl",          "rb"))
    sim   = pickle.load(open("models/similarity_compact.pkl", "rb"))
    return km, sc, c2s, rfm, sim

df, elbow_data           = load_data()
kmeans, scaler, c2s, rfm, sim_df = load_models()

ALL_PRODUCTS = sorted(sim_df.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            " Home",
            " Sales Analytics",
            " Country Analysis",
            " RFM Analysis",
            " Cluster Selection",
            " Customer Segmentation",
            " Similarity Matrix",
            " Product Recommendation",
            " Predict Segment",
            " Business Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Customer Segmentation &\nProduct Recommendation System")

# strip emoji prefix for logic
PAGE = page.split("  ", 1)[-1].strip()

# ── helpers ───────────────────────────────────────────────────────────────────
def section(title, subtitle=""):
    st.markdown(
        f"<h2 style='color:#0F766E;font-weight:700;margin-bottom:2px'>{title}</h2>",
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f"<p style='color:#71717A;margin-top:4px;margin-bottom:4px;font-size:14px'>{subtitle}</p>",
            unsafe_allow_html=True
        )
    st.markdown("---")

def kpi(col, label, value, delta=None):
    with col:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "Home":
    st.markdown("<h1 style='color:#0F766E;font-weight:800;letter-spacing:-1px'>Shopper Spectrum</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#71717A;font-size:16px;margin-top:-10px'>"
        "Customer Segmentation & Product Recommendation Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Customers",    f"{df['CustomerID'].nunique():,}")
    kpi(c2, "Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    kpi(c3, "Total Revenue",      f"£{df['TotalPrice'].sum():,.0f}")
    kpi(c4, "Unique Products",    f"{df['Description'].nunique():,}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='kpi-card'>
          <h4 style='margin:0 0 6px;color:#1C1C1E'>👤 Customer Segmentation</h4>
          <p style='color:#71717A;font-size:14px;margin:0'>
            Enter Recency, Frequency, and Monetary values to predict
            which segment a customer belongs to.</p>
          <p style='color:#0D9488;font-size:13px;margin:8px 0 0'>
            → Navigate to <b>Predict Segment</b></p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='kpi-card'>
          <h4 style='margin:0 0 6px;color:#1C1C1E'>🎯 Product Recommendation</h4>
          <p style='color:#71717A;font-size:14px;margin:0'>
            Enter a product name to discover 5 similar products
            based on purchase history patterns.</p>
          <p style='color:#0D9488;font-size:13px;margin:8px 0 0'>
            → Navigate to <b>Product Recommendation</b></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Customer Segments at a Glance")

    seg_summary = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
    seg_summary["Customers"] = rfm["Segment"].value_counts()
    seg_summary = seg_summary.reindex(["High-Value","Regular","Occasional","At-Risk"])
    seg_summary.columns = ["Avg Recency (days)","Avg Frequency","Avg Monetary (£)","Customers"]

    c1, c2, c3, c4 = st.columns(4)
    for col, seg in zip([c1,c2,c3,c4], ["High-Value","Regular","Occasional","At-Risk"]):
        row = seg_summary.loc[seg]
        color = SEG_COLORS[seg]
        icon  = SEG_ICONS[seg]
        with col:
            st.markdown(f"""
            <div class='seg-card' style='border-color:{color};background:{color}11'>
              <div style='font-weight:700;font-size:15px;color:{color}'>{icon} {seg}</div>
              <div style='font-size:22px;font-weight:800;margin:6px 0;color:#1C1C1E'>{int(row["Customers"])}</div>
              <div style='font-size:12px;color:#71717A'>customers</div>
              <hr style='border-color:#E7E5E4;margin:8px 0'>
              <div style='font-size:12px;color:#52525B'>
                R: {row["Avg Recency (days)"]}d &nbsp;|&nbsp;
                F: {row["Avg Frequency"]} &nbsp;|&nbsp;
                M: £{row["Avg Monetary (£)"]:.0f}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SALES ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Sales Analytics":
    section("📈 Sales Analytics", "Revenue trends, top products, and transaction distribution")

    # Monthly revenue
    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
    monthly.columns = ["Month","Revenue"]
    fig = px.area(monthly, x="Month", y="Revenue",
                  labels={"Revenue":"Revenue (£)","Month":""},
                  color_discrete_sequence=[TEAL])
    fig.update_traces(fill="tozeroy", line_color=TEAL, fillcolor="rgba(13,148,136,0.15)")
    fig_layout(fig, height=320, legend=False)
    fig.update_layout(title=dict(text="Monthly Revenue Trend", font=dict(color="#1C1C1E", size=14)))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top_qty = (df.groupby("Description")["Quantity"].sum()
                     .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_qty, x="Quantity", y="Description", orientation="h",
                     color_discrete_sequence=[TEAL],
                     labels={"Quantity":"Units Sold","Description":""})
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, title=dict(text="Top 10 by Quantity", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=380, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_rev = (df.groupby("Description")["TotalPrice"].sum()
                     .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_rev, x="TotalPrice", y="Description", orientation="h",
                     color_discrete_sequence=[DARK_TEAL],
                     labels={"TotalPrice":"Revenue (£)","Description":""})
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, title=dict(text="Top 10 by Revenue", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=380, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Transaction value distribution (remove top 1% outliers for readability)
    cap = df["TotalPrice"].quantile(0.99)
    fig = px.histogram(df[df["TotalPrice"] < cap], x="TotalPrice", nbins=60,
                       color_discrete_sequence=[TEAL],
                       labels={"TotalPrice":"Transaction Value (£)","count":"Frequency"})
    fig.update_layout(title=dict(text="Transaction Value Distribution (top 1% excluded)", font=dict(color="#1C1C1E", size=14)))
    fig_layout(fig, height=300, legend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Country Analysis":
    section("🌍 Country Analysis", "Revenue and transaction breakdown by country")

    country = df.groupby("Country").agg(
        Transactions=("InvoiceNo","nunique"),
        Revenue=("TotalPrice","sum"),
        Customers=("CustomerID","nunique"),
    ).sort_values("Revenue", ascending=False).reset_index()

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Countries",         f"{len(country)}")
    kpi(c2, "Top Country",       country.iloc[0]["Country"])
    kpi(c3, "Top Country Rev",   f"£{country.iloc[0]['Revenue']:,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(country.head(12), x="Revenue", y="Country", orientation="h",
                     color_discrete_sequence=[TEAL],
                     labels={"Revenue":"Revenue (£)","Country":""})
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, title=dict(text="Top 12 Countries by Revenue", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=420, legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(country.sort_values("Transactions", ascending=False).head(12),
                     x="Transactions", y="Country", orientation="h",
                     color_discrete_sequence=[DARK_TEAL],
                     labels={"Transactions":"Transactions","Country":""})
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, title=dict(text="Top 12 Countries by Transactions", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=420, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full Country Breakdown")
    st.dataframe(country.style.format({"Revenue":"£{:,.0f}"}),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RFM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "RFM Analysis":
    section("📊 RFM Analysis", "Distribution of Recency, Frequency, and Monetary values")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(rfm, x="Recency", nbins=40, color_discrete_sequence=[TEAL],
                           labels={"Recency":"Recency (days)"})
        fig.update_layout(title=dict(text="Recency Distribution", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=300, legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cap = rfm["Frequency"].quantile(0.99)
        fig = px.histogram(rfm[rfm["Frequency"] < cap], x="Frequency",
                           nbins=40, color_discrete_sequence=[DARK_TEAL],
                           labels={"Frequency":"Frequency (purchases)"})
        fig.update_layout(title=dict(text="Frequency Distribution", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=300, legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        cap = rfm["Monetary"].quantile(0.99)
        fig = px.histogram(rfm[rfm["Monetary"] < cap], x="Monetary",
                           nbins=40, color_discrete_sequence=[PURPLE],
                           labels={"Monetary":"Monetary (£)"})
        fig.update_layout(title=dict(text="Monetary Distribution", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=300, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Box plots by segment
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(rfm, x="Segment", y="Recency",
                     color="Segment", color_discrete_map=SEG_COLORS,
                     category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]})
        fig.update_layout(title=dict(text="Recency by Segment", font=dict(color="#1C1C1E", size=14)), showlegend=False)
        fig_layout(fig, height=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rfm_cap = rfm[rfm["Monetary"] < rfm["Monetary"].quantile(0.99)]
        fig = px.box(rfm_cap, x="Segment", y="Monetary",
                     color="Segment", color_discrete_map=SEG_COLORS,
                     category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]})
        fig.update_layout(title=dict(text="Monetary by Segment (top 1% excluded)", font=dict(color="#1C1C1E", size=14)), showlegend=False)
        fig_layout(fig, height=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(rfm[["Recency","Frequency","Monetary"]].describe().round(2),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTER SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Cluster Selection":
    section("📐 Cluster Selection", "Elbow Method and Silhouette Score used to pick K = 4")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow_data["K"], y=elbow_data["Inertia"],
            mode="lines+markers",
            line=dict(color=TEAL, width=3),
            marker=dict(size=9, color=TEAL, line=dict(width=2, color=CARD)),
            name="Inertia"
        ))
        # mark K=4
        k4_row = elbow_data[elbow_data["K"] == 4].iloc[0]
        fig.add_trace(go.Scatter(
            x=[k4_row["K"]], y=[k4_row["Inertia"]],
            mode="markers",
            marker=dict(size=14, color=CORAL, symbol="star"),
            name="Chosen K=4"
        ))
        fig.update_layout(title=dict(text="Elbow Method — Inertia vs K", font=dict(color="#1C1C1E", size=14)),
                          xaxis_title=dict(text="Number of Clusters (K)", font=dict(color="#1C1C1E", size=14)),
                          yaxis_title=dict(text="Inertia", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The 'elbow' is where inertia stops improving significantly — beyond K=4, gains are marginal.")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow_data["K"], y=elbow_data["Silhouette"],
            mode="lines+markers",
            line=dict(color=AMBER, width=3),
            marker=dict(size=9, color=AMBER, line=dict(width=2, color=CARD)),
            name="Silhouette"
        ))
        fig.add_trace(go.Scatter(
            x=[k4_row["K"]], y=[elbow_data[elbow_data["K"]==4]["Silhouette"].values[0]],
            mode="markers",
            marker=dict(size=14, color=CORAL, symbol="star"),
            name="Chosen K=4"
        ))
        fig.update_layout(title=dict(text="Silhouette Score vs K", font=dict(color="#1C1C1E", size=14)),
                          xaxis_title=dict(text="Number of Clusters (K)", font=dict(color="#1C1C1E", size=14)),
                          yaxis_title=dict(text="Silhouette Score", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Silhouette Score measures how well-separated clusters are. Closer to 1.0 is better.")

    st.info(
        "**K = 4 was selected** — it aligns with the four business segments required: "
        "High-Value, Regular, Occasional, and At-Risk. While K=5 gave marginally higher "
        "silhouette, the business requirement takes priority."
    )

    st.markdown("#### Elbow Data Table")
    st.dataframe(elbow_data.style.format({"Inertia":"{:,.1f}","Silhouette":"{:.4f}"}),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Customer Segmentation":
    section("👥 Customer Segmentation", "KMeans-derived segments visualised in 2D and 3D")

    c1, c2 = st.columns([1, 2])
    with c1:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig = px.pie(seg_counts, names="Segment", values="Count",
                     color="Segment", color_discrete_map=SEG_COLORS,
                     hole=0.45)
        fig.update_traces(textinfo="percent", textfont_size=11, textposition="inside")
        fig.update_layout(showlegend=True)
        fig.update_layout(title=dict(text="Segment Distribution", font=dict(color="#1C1C1E", size=14)), showlegend=False)
        fig_layout(fig, height=360, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            rfm, x="Frequency", y="Monetary",
            color="Segment", color_discrete_map=SEG_COLORS,
            size="Monetary", size_max=22, opacity=0.75,
            hover_data={"CustomerID":True,"Recency":True,"Frequency":True,"Monetary":":.0f"},
            category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]},
        )
        fig.update_layout(title=dict(text="Customer Clusters — Frequency vs Monetary", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # 3-D scatter
    fig = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", color_discrete_map=SEG_COLORS,
        opacity=0.7, size_max=6,
        hover_data={"CustomerID":True},
        category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]},
    )
    fig.update_layout(
        title=dict(text="3D Customer Segmentation", font=dict(color="#1C1C1E", size=14)),
        scene=dict(
            xaxis=dict(backgroundcolor=BG, gridcolor="#E7E5E4",
                       tickfont=dict(color="#52525B"), title_font=dict(color="#52525B")),
            yaxis=dict(backgroundcolor=BG, gridcolor="#E7E5E4",
                       tickfont=dict(color="#52525B"), title_font=dict(color="#52525B")),
            zaxis=dict(backgroundcolor=BG, gridcolor="#E7E5E4",
                       tickfont=dict(color="#52525B"), title_font=dict(color="#52525B")),
        ),
        height=580,
        paper_bgcolor=BG,
        margin=dict(l=0,r=0,t=40,b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Segment profile cards
    st.markdown("#### Segment Profiles")
    seg_profiles = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
    cols = st.columns(4)
    for col, seg in zip(cols, ["High-Value","Regular","Occasional","At-Risk"]):
        row = seg_profiles.loc[seg]
        color = SEG_COLORS[seg]
        icon  = SEG_ICONS[seg]
        with col:
            st.markdown(f"""
            <div class='seg-card' style='border-color:{color};background:{color}0D'>
              <div style='font-weight:700;color:{color};font-size:14px'>{icon} {seg}</div>
              <div style='font-size:12px;color:#52525B;margin-top:8px'>
                <b>Recency:</b> {row.Recency}d<br>
                <b>Frequency:</b> {row.Frequency}<br>
                <b>Monetary:</b> £{row.Monetary:.0f}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMILARITY MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Similarity Matrix":
    section("🔥 Similarity Matrix",
            "Top 20 most similar products for any selected product — based on cosine similarity")

    selected_product = st.selectbox(
        "Select a product to explore its similarity scores",
        options=ALL_PRODUCTS,
        index=0,
    )

    if selected_product and selected_product in sim_df:
        top_similar = sim_df[selected_product]  # list of (product, score) tuples
        names  = [x[0] for x in top_similar]
        scores = [x[1] for x in top_similar]

        sim_table = pd.DataFrame({"Product": names, "Similarity Score": scores})

        # Bar chart of top 20
        fig = px.bar(
            sim_table, x="Similarity Score", y="Product", orientation="h",
            color="Similarity Score",
            color_continuous_scale=["#F0FDFA", TEAL_LIGHT, TEAL, DARK_TEAL],
            range_color=[0, 1],
            labels={"Similarity Score": "Cosine Similarity", "Product": ""},
        )
        fig.update_layout(
            title=dict(text=f"Top 20 Products Similar to: {selected_product[:50]}",
                       font=dict(color="#27272A", size=14)),
            yaxis=dict(categoryorder="total ascending",
                       tickfont=dict(color="#52525B", size=10),
                       title_font=dict(color="#52525B")),
            xaxis=dict(tickfont=dict(color="#52525B"), title_font=dict(color="#52525B")),
            coloraxis_showscale=True,
            height=550,
            paper_bgcolor=BG,
            plot_bgcolor=CARD,
            margin=dict(l=16, r=16, t=50, b=16),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Full Similarity Table")
        st.dataframe(
            sim_table.style.format({"Similarity Score": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )
        st.caption(
            "Similarity score ranges from 0 (no overlap) to 1 (identical purchase pattern). "
            "Two products score high when the same customers frequently buy both."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Product Recommendation":
    section("🎯 Product Recommendation",
            "Item-based collaborative filtering — powered by cosine similarity")

    product_input = st.selectbox(
        "Search and select a product",
        options=ALL_PRODUCTS,
        index=None,
        placeholder="Start typing a product name...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run = st.button("Recommend Products", use_container_width=True)

    if run:
        if product_input:
            # compact dict: list of (name, score) tuples, already sorted
            scores = sim_df[product_input][:5]  # top 5 from precomputed list
            st.markdown(f"#### Top 5 products similar to: *{product_input}*")
            st.markdown("---")
            cols = st.columns(5)
            for i, (name, score) in enumerate(scores):
                bar_pct = int(score * 100)
                with cols[i]:
                    st.markdown(f"""
                    <div class='kpi-card' style='text-align:center;min-height:160px'>
                      <div style='font-size:28px;margin-bottom:8px'>🛍️</div>
                      <div style='font-size:12px;font-weight:600;color:#1C1C1E;
                                  min-height:48px;line-height:1.4'>{name}</div>
                      <div style='margin-top:10px;font-size:13px;
                                  color:{TEAL};font-weight:700'>{score:.2f}</div>
                      <div style='font-size:11px;color:#71717A'>similarity</div>
                      <div style='background:#E7E5E4;border-radius:4px;
                                  height:5px;margin-top:8px'>
                        <div style='background:{TEAL};width:{bar_pct}%;
                                    height:5px;border-radius:4px'></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("Please select a product first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SEGMENT
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Predict Segment":
    section("🔮 Predict Customer Segment",
            "Enter RFM values to classify a customer into a segment")

    c1, c2, c3, _ = st.columns([1, 1, 1, 1])
    with c1:
        recency  = st.number_input("Recency (days since last purchase)",
                                   min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (number of purchases)",
                                    min_value=1, value=5, step=1)
    with c3:
        monetary  = st.number_input("Monetary (total spend £)",
                                    min_value=0.0, value=500.0, step=10.0)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict = st.button("Predict Segment", use_container_width=True)

    if predict:
        inp = pd.DataFrame([[recency, frequency, monetary]],
                           columns=["Recency","Frequency","Monetary"])
        cluster = kmeans.predict(scaler.transform(inp))[0]
        segment = c2s[cluster]
        color   = SEG_COLORS[segment]
        icon    = SEG_ICONS[segment]

        desc = {
            "High-Value":  "Frequent, recent, and big spenders. Top priority for loyalty rewards.",
            "Regular":     "Steady buyers but not premium. Good candidates for upselling.",
            "Occasional":  "Rare purchases. Consider re-engagement or win-back campaigns.",
            "At-Risk":     "Haven't purchased in a long time. Needs urgent retention effort.",
        }

        st.markdown("---")
        st.markdown(f"""
        <div class='seg-card' style='border-color:{color};background:{color}0D;padding:24px 28px'>
          <div style='font-size:13px;color:#71717A;margin-bottom:4px'>
              Predicted Segment
          </div>
          <div style='font-size:32px;font-weight:800;color:{color};margin-bottom:6px'>
              {icon} {segment}
          </div>
          <div style='font-size:15px;color:#1C1C1E'>{desc[segment]}</div>
          <hr style='border-color:{color}33;margin:14px 0'>
          <div style='font-size:13px;color:#52525B'>
            Input — Recency: <b>{recency}d</b> &nbsp;|&nbsp;
            Frequency: <b>{frequency}</b> &nbsp;|&nbsp;
            Monetary: <b>£{monetary:,.0f}</b>
          </div>
        </div>""", unsafe_allow_html=True)

        # show where this customer sits relative to the segment averages
        st.markdown("#### How this customer compares to segment averages")
        seg_avg = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
        compare = seg_avg.copy()
        compare.loc["Your Input"] = [recency, frequency, monetary]
        compare = compare.reset_index().rename(columns={"index":"Segment"})
        compare["IsInput"] = compare["Segment"] == "Your Input"

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Recency (lower=better)",
                                            "Frequency (higher=better)",
                                            "Monetary £ (higher=better)"))
        for i, col_name in enumerate(["Recency","Frequency","Monetary"], 1):
            colors_bar = [CORAL if r else (SEG_COLORS.get(s, SLATE))
                          for s, r in zip(compare["Segment"], compare["IsInput"])]
            fig.add_trace(
                go.Bar(x=compare["Segment"], y=compare[col_name],
                       marker_color=colors_bar, showlegend=False),
                row=1, col=i
            )
        fig.update_layout(height=320, paper_bgcolor=BG, plot_bgcolor=CARD,
                          margin=dict(l=16,r=16,t=40,b=16),
                          font=dict(color="#52525B"))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Business Insights":
    section("💡 Business Insights",
            "Key findings derived directly from the cleaned dataset")

    total_rev   = df["TotalPrice"].sum()
    avg_spend   = rfm["Monetary"].mean()
    top_country = df.groupby("Country")["TotalPrice"].sum().idxmax()
    best_prod   = df.groupby("Description")["TotalPrice"].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Revenue",      f"£{total_rev:,.0f}")
    kpi(c2, "Avg Customer Spend", f"£{avg_spend:,.0f}")
    kpi(c3, "Top Country",        top_country)
    kpi(c4, "Best Selling Product", best_prod[:28]+"…" if len(best_prod)>28 else best_prod)

    st.markdown("---")
    seg_rev   = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    seg_pct   = (seg_rev / seg_rev.sum() * 100).round(1)
    seg_count = rfm["Segment"].value_counts()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            seg_rev.reset_index(), x="Segment", y="Monetary",
            color="Segment", color_discrete_map=SEG_COLORS,
            category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]},
            labels={"Monetary":"Total Revenue (£)"},
            text_auto=".2s",
        )
        fig.update_layout(title=dict(text="Revenue Contribution by Segment", font=dict(color="#1C1C1E", size=14)), showlegend=False)
        fig_layout(fig, height=340, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly_seg = df.merge(
            rfm[["CustomerID","Segment"]], on="CustomerID", how="left"
        )
        monthly_seg = (
            monthly_seg.set_index("InvoiceDate")
            .groupby("Segment")
            .resample("ME")["TotalPrice"].sum()
            .reset_index()
        )
        fig = px.line(monthly_seg, x="InvoiceDate", y="TotalPrice",
                      color="Segment", color_discrete_map=SEG_COLORS,
                      labels={"TotalPrice":"Revenue (£)","InvoiceDate":""},
                      category_orders={"Segment":["High-Value","Regular","Occasional","At-Risk"]})
        fig.update_layout(title=dict(text="Monthly Revenue by Segment", font=dict(color="#1C1C1E", size=14)))
        fig_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Key Findings")
    insights = [
        (TEAL,   "💎", f"<b>High-Value segment</b> has only <b>{seg_count.get('High-Value', 0)} customers</b> "
                       f"but contributes <b>{seg_pct.get('High-Value', 0)}%</b> of total revenue — "
                       f"a clear target for loyalty and retention programmes."),
        (AMBER,  "📦", f"<b>Occasional shoppers</b> are the largest group "
                       f"({seg_count.get('Occasional', 0)} customers, "
                       f"{seg_pct.get('Occasional', 0)}% of revenue) — "
                       f"small nudges (email, discounts) could convert some to Regular buyers."),
        (CORAL,  "⚠️", f"<b>{seg_count.get('At-Risk', 0)} At-Risk customers</b> haven't purchased recently. "
                       f"Win-back campaigns targeting this group could recover "
                       f"£{seg_rev.get('At-Risk', 0):,.0f} in potential revenue."),
        (PURPLE, "🌍", f"<b>{top_country}</b> dominates with "
                       f"{df[df['Country']==top_country]['InvoiceNo'].nunique():,} transactions, "
                       f"accounting for the vast majority of revenue."),
        (SLATE,  "🔄", f"<b>Regular customers</b> ({seg_count.get('Regular', 0)} customers) represent a strong "
                       f"middle tier — upselling via personalised recommendations could move "
                       f"a portion of them toward High-Value status."),
    ]
    for color, icon, text in insights:
        st.markdown(
            f"<div class='insight-row' style='border-left:4px solid {color}'>"
            f"{icon} {text}</div>",
            unsafe_allow_html=True,
        )