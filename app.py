import os
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ─────────────────────────────────────────────────────────────
# 1) LOAD — only the columns we actually need
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output_chunks"
NEEDED_COLS = ["title", "author", "genre", "bookformat",
               "rating", "pages", "totalratings"]

part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "output_part_*.csv")))
if not part_files:
    raise FileNotFoundError(
        f"No CSV parts found in '{OUTPUT_DIR}/'. "
        "Make sure output_part_1.csv … output_part_N.csv exist."
    )

print(f"📂 Loading {len(part_files)} file(s) …")

chunks = []
for f in part_files:
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"   📦 {os.path.basename(f)} — {size:.2f} MB")
    tmp = pd.read_csv(f, usecols=lambda c: c.lower() in NEEDED_COLS,
                      low_memory=True)
    tmp.columns = tmp.columns.str.lower()
    chunks.append(tmp)

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"✅ Loaded {len(df):,} rows")

# ─────────────────────────────────────────────────────────────
# 2) CLEANING + MEMORY-EFFICIENT DTYPES
# ─────────────────────────────────────────────────────────────
df["title"]  = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()

df["genre"]      = df["genre"].astype(str).str.split(",").str[0].str.strip()
df["bookformat"] = df["bookformat"].astype(str).str.strip().str.title()

df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["pages"]        = pd.to_numeric(df["pages"],        errors="coerce")

df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()

# Downcast numerics to save ~40 % RAM
df["rating"]       = df["rating"].astype("float32")
df["pages"]        = df["pages"].astype("float32")
df["totalratings"] = df["totalratings"].astype("float32")

# Category columns — huge RAM win on repeated strings
df["bookformat"] = df["bookformat"].astype("category")
df["genre"]      = df["genre"].astype("category")

top_formats  = df["bookformat"].value_counts().head(6).index.tolist()
top3_formats = top_formats[:3]
top4_formats = top_formats[:4]
top8_genres  = df["genre"].value_counts().head(8).index.tolist()

print(f"🎉 Dataset ready: {len(df):,} rows | {df.shape[1]} cols | "
      f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ─────────────────────────────────────────────────────────────
# 3) PRE-AGGREGATE — build small summary tables once at startup
#    so figures never touch the full df again
# ─────────────────────────────────────────────────────────────

# --- Fig 1 summary (histogram bins, not raw rows) -------------
RATING_BINS = np.arange(0, 5.15, 0.125)          # 40-bin edges
hist_counts, bin_edges = np.histogram(df["rating"].dropna(), bins=RATING_BINS)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
mean_r   = float(df["rating"].mean())
median_r = float(df["rating"].median())

# --- Fig 2 summary (10 rows) ----------------------------------
top10_fmt = df["bookformat"].value_counts().head(10).reset_index()
top10_fmt.columns = ["bookformat", "count"]
top10_fmt["share"]      = (top10_fmt["count"] / len(df) * 100).round(1)
top10_fmt["avg_rating"] = top10_fmt["bookformat"].map(
    df.groupby("bookformat", observed=True)["rating"].mean().round(2)
)

# --- Fig 3 sample (2 000 rows) --------------------------------
scat3_df = (df[df["bookformat"].isin(top_formats)]
            .sample(min(2000, len(df)), random_state=42)
            [["title", "author", "pages", "rating", "totalratings", "bookformat"]]
            .copy())
scat3_df["log_tr"] = np.log1p(scat3_df["totalratings"]).astype("float32")

# --- Fig 4 summary (histogram bins per format) ----------------
fig4_data = {}
for fmt in top3_formats:
    vals = df.loc[df["bookformat"] == fmt, "rating"].dropna().values
    counts, edges = np.histogram(vals, bins=np.arange(0, 5.15, 0.15))
    centers = (edges[:-1] + edges[1:]) / 2
    fig4_data[fmt] = {"centers": centers, "counts": counts,
                      "mean": float(vals.mean())}

# --- Fig 5 summary (aggregated, tiny) -------------------------
df_anim = (df[df["genre"].isin(top8_genres) & df["bookformat"].isin(top4_formats)]
           .groupby(["bookformat", "genre"], observed=True, as_index=False)
           .agg(count=("rating", "count"), avg_rating=("rating", "mean"))
           .round({"avg_rating": 3}))
genre_order = (df_anim.groupby("genre")["count"].sum()
               .sort_values(ascending=False).index.tolist())

print("✅ Pre-aggregations done")

# ─────────────────────────────────────────────────────────────
# 4) BUILD STATIC FIGURES FROM SUMMARIES (not from full df)
# ─────────────────────────────────────────────────────────────

# Fig 1 ── Rating histogram (from bin counts, ~40 rows)
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=bin_centers, y=hist_counts,
    marker_color="#4C72B0", opacity=0.85,
    hovertemplate="Rating: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
    name="",
))
fig1.add_vline(x=mean_r,   line_color="crimson", line_dash="dash",
               annotation_text=f"Mean {mean_r:.2f}",
               annotation_position="top right")
fig1.add_vline(x=median_r, line_color="green",   line_dash="dot",
               annotation_text=f"Median {median_r:.2f}",
               annotation_position="top left")
fig1.update_layout(
    title="Distribution of Book Ratings — GoodReads 100k",
    title_font_size=16,
    xaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#e0e0e0"),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#e0e0e0"),
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    bargap=0.05, showlegend=False,
    font=dict(family="Arial", size=13),
    annotations=[dict(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text="Ratings are left-skewed — most books cluster 3.5–4.2",
        showarrow=False, bgcolor="lightyellow", bordercolor="gray",
        font=dict(size=11, color="navy"),
    )],
)

# Fig 2 ── Horizontal bar (10 rows)
fig2 = px.bar(
    top10_fmt, x="count", y="bookformat", orientation="h",
    color="count", color_continuous_scale="Blues",
    title="Top 10 Book Formats by Count",
    labels={"count": "Number of Books", "bookformat": ""},
    text="count", custom_data=["share", "avg_rating"], height=420,
)
fig2.update_traces(
    texttemplate="%{x:,}", textposition="outside",
    hovertemplate="<b>%{y}</b><br>Books: %{x:,}<br>Share: %{customdata[0]:.1f}%"
                  "<br>Avg Rating: %{customdata[1]:.2f}<extra></extra>",
)
fig2.update_layout(
    yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    xaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    yaxis_title="", title_font_size=16, height=420,
)

# Fig 3 ── Bubble scatter (2 000 sampled rows)
fig3 = px.scatter(
    scat3_df, x="pages", y="rating",
    color="bookformat", size="log_tr", size_max=16, opacity=0.55,
    hover_data={"title": True, "author": True, "pages": True,
                "rating": ":.2f", "totalratings": True, "log_tr": False},
    labels={"pages": "Pages", "rating": "Rating",
            "bookformat": "Format", "totalratings": "Total Ratings"},
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Pages vs. Rating — Bubble Size = log(Total Ratings)",
)
fig3.update_layout(
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    title_font_size=16, legend_title="Format",
    xaxis=dict(title="Pages",        showgrid=True, gridcolor="#ddd"),
    yaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#ddd"),
)
del scat3_df   # free immediately after use

# Fig 4 ── Multi-trace overlay histogram (from bin counts)
COLORS4 = ["#4C72B0", "#DD8452", "#55A868"]
fig4 = go.Figure()
for fmt, color in zip(top3_formats, COLORS4):
    d = fig4_data[fmt]
    fig4.add_trace(go.Bar(
        x=d["centers"], y=d["counts"], name=fmt,
        opacity=0.65, marker_color=color,
        hovertemplate=f"<b>{fmt}</b><br>Rating: %{{x:.2f}}<br>Books: %{{y:,}}<extra></extra>",
    ))
    fig4.add_vline(x=d["mean"], line_color=color, line_dash="dash", line_width=2,
                   annotation_text=f"{fmt} mean {d['mean']:.2f}",
                   annotation_font_color=color, annotation_font_size=10)
fig4.update_layout(
    barmode="overlay",
    title="Rating Distribution: Paperback vs Hardcover vs Ebook (Multi-Trace)",
    title_font_size=16,
    xaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#ddd"),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    legend=dict(title="Format", x=0.02, y=0.97,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray"),
    annotations=[dict(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text="Click legend items to show/hide individual traces",
        showarrow=False, font=dict(size=11, color="gray"),
    )],
    height=480,
)

# Fig 5 ── Animated bar (tiny aggregated df)
fig5 = px.bar(
    df_anim, x="genre", y="count",
    animation_frame="bookformat", color="genre",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title="Top 8 Genres by Book Count — Animated by Format",
    labels={"genre": "Genre", "count": "Number of Books",
            "avg_rating": "Avg Rating", "bookformat": "Format"},
    hover_data={"count": True, "avg_rating": ":.3f", "bookformat": False},
    category_orders={"genre": genre_order},
    text="count",
)
fig5.update_traces(
    texttemplate="%{y:,}", textposition="outside",
    hovertemplate="<b>%{x}</b><br>Books: %{y:,}<br>Avg Rating: %{customdata[0]:.2f}<extra></extra>",
)
fig5.update_layout(
    title_font_size=16, plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    xaxis=dict(title="Genre", showgrid=False, tickangle=-20),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    showlegend=False, height=500,
    annotations=[dict(
        x=0.5, y=-0.18, xref="paper", yref="paper",
        text="▶ Press Play — each frame shows genre counts for one book format",
        showarrow=False, font=dict(size=11, color="gray"),
    )],
)
if fig5.layout.updatemenus:
    fig5.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 1200
    fig5.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

print("✅ All 5 figures built")

# ─────────────────────────────────────────────────────────────
# 5) DASH APP
# ─────────────────────────────────────────────────────────────
app  = Dash(__name__, suppress_callback_exceptions=True)
server = app.server   # REQUIRED FOR RENDER

TAB_STYLE     = {"fontFamily": "Arial", "fontSize": "14px", "padding": "10px 20px"}
TAB_SEL_STYLE = {**TAB_STYLE, "borderTop": "3px solid #1a3c6b",
                 "fontWeight": "bold", "color": "#1a3c6b"}
CARD_STYLE    = {
    "background": "white", "borderRadius": "8px",
    "padding": "12px 20px", "boxShadow": "0 1px 6px rgba(0,0,0,0.1)",
    "minWidth": "130px", "flex": "1",
}
CHART_CARD    = {
    "background": "white", "borderRadius": "8px", "padding": "16px",
    "boxShadow": "0 1px 6px rgba(0,0,0,0.1)", "flex": "1", "minWidth": "360px",
}

def section_title(text):
    return html.H3(text, style={
        "color": "#1a3c6b", "borderLeft": "4px solid #1a3c6b",
        "paddingLeft": "10px", "marginBottom": "8px", "fontSize": "15px",
    })

# ── Tab 1: Notebook figures ───────────────────────────────────
tab_figures = html.Div([html.Div([
    html.P("All 5 interactive Plotly figures from Phase 2 (Task 1).",
           style={"color": "#555", "fontSize": "13px", "margin": "0 0 20px"}),
    *[html.Div([section_title(t), dcc.Graph(figure=f, config={"displayModeBar": True})],
               style={"marginBottom": "24px"})
      for t, f in [
          ("Fig 1 — Rating Distribution (Histogram + Violin)", fig1),
          ("Fig 2 — Top 10 Book Formats (Horizontal Bar)",     fig2),
          ("Fig 3 — Pages vs. Rating (Bubble Scatter)",        fig3),
          ("Fig 4 — Rating by Format (Multi-Trace Histogram)", fig4),
          ("Fig 5 — Top Genres Animated by Format",            fig5),
      ]],
], style={"padding": "24px 30px", "background": "#f0f4fb", "minHeight": "100vh"})])

# ── Tab 2: Interactive dashboard ──────────────────────────────
tab_dashboard = html.Div([
    html.Div([
        html.Div([
            html.Label("Book Format",
                       style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="format-dropdown",
                options=[{"label": f, "value": f} for f in top_formats],
                value=top_formats[:3], multi=True,
                style={"fontSize": "13px"},
            ),
        ], style={"flex": "1", "minWidth": "280px"}),
        html.Div([
            html.Label("Rating Range",
                       style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RangeSlider(
                id="rating-slider", min=0, max=5, step=0.1,
                value=[3.0, 5.0], marks={i: str(i) for i in range(6)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"flex": "2", "minWidth": "300px"}),
    ], style={
        "display": "flex", "gap": "24px", "flexWrap": "wrap",
        "padding": "20px 30px", "background": "white",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.07)", "alignItems": "flex-end",
    }),
    html.Div(id="kpi-row", style={
        "display": "flex", "gap": "16px",
        "padding": "16px 30px", "background": "#fafbff", "flexWrap": "wrap",
    }),
    html.Div([
        html.Div([
            html.H3("Top Genres by Book Count",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="genre-bar", config={"displayModeBar": True}),
        ], style=CHART_CARD),
        html.Div([
            html.H3("Pages vs. Rating",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="scatter-chart", config={"displayModeBar": True}),
        ], style=CHART_CARD),
    ], style={"display": "flex", "gap": "16px",
              "padding": "16px 30px", "flexWrap": "wrap"}),
], style={"background": "#f0f4fb", "minHeight": "100vh"})

# ── Full layout ───────────────────────────────────────────────
app.layout = html.Div([
    html.Div([
        html.H1("📚 GoodReads Dashboard",
                style={"margin": "0", "color": "white", "fontSize": "22px"}),
        html.P("Phase 2 — Interactive Analytics | GoodReads 100k Books",
               style={"margin": "4px 0 0", "color": "#cfe2ff", "fontSize": "13px"}),
    ], style={"background": "#1a3c6b", "padding": "20px 30px"}),

    dcc.Tabs(id="main-tabs", value="tab-figures", children=[
        dcc.Tab(label="📊 Notebook Figures (Task 1)",    value="tab-figures",
                style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
        dcc.Tab(label="🔍 Interactive Dashboard (Task 2)", value="tab-dashboard",
                style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
    ]),
    html.Div(id="tab-content"),
    html.Div(
        "Student: Nour Eddin Abuazzam | ID: 202210576 | GoodReads 100k Dataset",
        style={"textAlign": "center", "padding": "14px", "color": "#666",
               "fontSize": "12px", "borderTop": "1px solid #ddd", "background": "white"},
    ),
], style={"fontFamily": "Arial", "background": "#f0f4fb", "minHeight": "100vh"})

# ─────────────────────────────────────────────────────────────
# 6) CALLBACKS
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    return tab_figures if tab == "tab-figures" else tab_dashboard


@app.callback(
    Output("genre-bar",     "figure"),
    Output("scatter-chart", "figure"),
    Output("kpi-row",       "children"),
    Input("format-dropdown", "value"),
    Input("rating-slider",   "value"),
)
def update(selected_formats, rating_range):
    if not selected_formats:
        selected_formats = top_formats[:3]
    low, high = rating_range

    # Work with a tiny boolean mask — never copy the full df
    mask = (
        df["bookformat"].isin(selected_formats) &
        (df["rating"] >= low) &
        (df["rating"] <= high)
    )
    filtered = df.loc[mask]

    # KPIs
    def kpi_card(label, value, color="#1a3c6b"):
        return html.Div([
            html.P(label,  style={"margin": "0", "fontSize": "12px", "color": "#666"}),
            html.H2(value, style={"margin": "4px 0 0", "color": color, "fontSize": "22px"}),
        ], style=CARD_STYLE)

    n = len(filtered)
    avg_rating   = round(float(filtered["rating"].mean()), 2) if n else 0
    median_pages = int(filtered["pages"].median())             if n else 0
    kpis = [
        kpi_card("Books in Selection", f"{n:,}",               "#1a3c6b"),
        kpi_card("Avg Rating",         f"{avg_rating:.2f} ⭐", "#2e7d32"),
        kpi_card("Median Pages",       f"{median_pages:,}",    "#6a1b9a"),
        kpi_card("Formats Selected",   str(len(selected_formats)), "#c62828"),
    ]

    # Genre bar — aggregate only, no copy
    gc = filtered["genre"].value_counts().head(10).reset_index()
    gc.columns = ["Genre", "Books"]
    gc["avg_rating"] = gc["Genre"].map(
        filtered.groupby("genre", observed=True)["rating"].mean().round(3)
    )
    bar_fig = px.bar(
        gc, x="Books", y="Genre", orientation="h",
        color="Books", color_continuous_scale="Blues",
        text="Books", custom_data=["Genre", "avg_rating"],
    )
    bar_fig.update_traces(
        texttemplate="%{x:,}", textposition="outside",
        hovertemplate="<b>%{y}</b><br>Books: %{x:,}"
                      "<br>Avg Rating: %{customdata[1]:.2f}<extra></extra>",
    )
    bar_fig.update_layout(
        yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
        plot_bgcolor="#f9f9f9", margin=dict(l=10, r=10, t=10, b=10), height=380,
    )

    # Scatter — small sample, only needed columns
    if n > 0:
        samp = filtered[["title", "author", "pages", "rating",
                          "totalratings", "bookformat"]].sample(
            min(2000, n), random_state=42
        ).copy()
        samp["log_tr"] = np.log1p(samp["totalratings"]).astype("float32")
        scatter = px.scatter(
            samp, x="pages", y="rating",
            color="bookformat", size="log_tr", size_max=16, opacity=0.55,
            hover_data={"title": True, "author": True, "pages": True,
                        "rating": ":.2f", "totalratings": True, "log_tr": False},
            labels={"pages": "Pages", "rating": "Rating",
                    "bookformat": "Format", "totalratings": "Total Ratings"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        scatter.update_layout(
            plot_bgcolor="#f9f9f9", legend_title="Format",
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        del samp
    else:
        scatter = px.scatter(title="No data for selected filters")

    return bar_fig, scatter, kpis


# ─────────────────────────────────────────────────────────────
# 7) RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
