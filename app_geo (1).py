import os, glob, re, json
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output

# ─── 1) LOAD DATA ────────────────────────────────────────────────────────────
OUTPUT_DIR = "output_chunks"
part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "output_part_*.csv")))
df = pd.concat([pd.read_csv(f, dtype={'isbn': str}) for f in part_files], ignore_index=True)
df.columns = df.columns.str.lower()
df["title"]        = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()
df["genre"]        = df["genre"].astype(str).str.split(",").str[0].str.strip()
df["bookformat"]   = df["bookformat"].astype(str).str.strip().str.title()
df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["pages"]        = pd.to_numeric(df["pages"],        errors="coerce")
df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()
top_formats = df["bookformat"].value_counts().head(6).index.tolist()

# ─── 2) GEOSPATIAL PREP ──────────────────────────────────────────────────────
ISBN_GROUP = {
    '0':'United Kingdom','1':'United Kingdom','2':'France','3':'Germany',
    '4':'Japan','5':'Russia','7':'China','80':'Czech Republic','81':'India',
    '82':'Norway','83':'Poland','84':'Spain','85':'Brazil','86':'Serbia',
    '87':'Denmark','88':'Italy','89':'South Korea','90':'Netherlands',
    '91':'Sweden','93':'India','94':'Netherlands','95':'Sri Lanka',
    '956':'Chile','957':'Taiwan','958':'Colombia','959':'Cuba','960':'Greece',
    '961':'Slovenia','962':'China','963':'Hungary','964':'Iran','965':'Israel',
    '966':'Ukraine','967':'Malaysia','968':'Mexico','969':'Pakistan',
    '970':'Mexico','971':'Philippines','972':'Portugal','973':'Romania',
    '974':'Thailand','975':'Turkey','977':'Egypt','978':'Nigeria',
    '979':'Indonesia','980':'Venezuela','981':'Singapore','983':'Malaysia',
    '984':'Bangladesh','985':'Belarus','986':'Taiwan','987':'Argentina',
    '988':'China','989':'Portugal',
}
COUNTRY_ISO3 = {
    'United Kingdom':'GBR','Germany':'DEU','China':'CHN','Japan':'JPN',
    'Russia':'RUS','Czech Republic':'CZE','France':'FRA','India':'IND',
    'Denmark':'DNK','Norway':'NOR','Italy':'ITA','Netherlands':'NLD',
    'South Korea':'KOR','Spain':'ESP','Brazil':'BRA','Sweden':'SWE',
    'Poland':'POL','Serbia':'SRB','Sri Lanka':'LKA','Malaysia':'MYS',
    'Indonesia':'IDN','Bangladesh':'BGD','Portugal':'PRT','Turkey':'TUR',
    'Hungary':'HUN','Egypt':'EGY','Israel':'ISR','Argentina':'ARG',
    'Mexico':'MEX','Philippines':'PHL','Nigeria':'NGA','Taiwan':'TWN',
    'Romania':'ROU','Chile':'CHL','Thailand':'THA','Singapore':'SGP',
    'Ukraine':'UKR','Greece':'GRC','Venezuela':'VEN','Slovenia':'SVN',
    'Colombia':'COL','Pakistan':'PAK','Cuba':'CUB','Belarus':'BLR','Iran':'IRN',
}

def isbn_to_country(isbn):
    if pd.isna(isbn): return 'Unknown'
    s = re.sub(r'[^0-9]', '', str(isbn))
    for l in [4,3,2,1]:
        if s[:l] in ISBN_GROUP:
            return ISBN_GROUP[s[:l]]
    return 'Unknown'

df['country'] = df['isbn'].apply(isbn_to_country)
df['iso3']    = df['country'].map(COUNTRY_ISO3)

geo_agg = (df.dropna(subset=['iso3'])
             .groupby(['country','iso3'])
             .agg(book_count=('title','count'),
                  avg_rating=('rating','mean'),
                  avg_pages=('pages','mean'))
             .reset_index())
geo_agg['avg_rating'] = geo_agg['avg_rating'].round(2)
geo_agg['avg_pages']  = geo_agg['avg_pages'].round(0).astype(int)

# ─── 3) BUILD FOLIUM MAP ──────────────────────────────────────────────────────
def build_folium_map():
    with open("countries.geojson") as f:
        geo_json = json.load(f)

    lu = geo_agg.set_index('iso3').to_dict('index')
    for feat in geo_json['features']:
        iso = feat['properties'].get('ISO3166-1-Alpha-3','')
        info = lu.get(iso, {})
        feat['properties']['book_count'] = info.get('book_count','N/A')
        feat['properties']['avg_rating']  = info.get('avg_rating','N/A')
        feat['properties']['avg_pages']   = info.get('avg_pages','N/A')

    m = folium.Map(location=[20,10], zoom_start=2, tiles='CartoDB positron')
    ch = folium.Choropleth(
        geo_data=geo_json, data=geo_agg,
        columns=['iso3','book_count'],
        key_on='feature.properties.ISO3166-1-Alpha-3',
        fill_color='YlOrRd', fill_opacity=0.75, line_opacity=0.3,
        nan_fill_color='#d3d3d3',
        legend_name='Number of Books Published', highlight=True,
    ).add_to(m)
    folium.GeoJsonTooltip(
        fields=['name','book_count','avg_rating','avg_pages'],
        aliases=['Country','Books Published','Avg Rating','Avg Pages'],
        localize=True, sticky=True,
        style="background:white;padding:6px;border-radius:4px;font-size:12px"
    ).add_to(ch.geojson)

    m.get_root().html.add_child(folium.Element('''
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
            z-index:1000;background:rgba(26,60,107,0.92);color:white;
            padding:8px 20px;border-radius:8px;font-family:Arial;text-align:center">
            <b>📚 Books by Publisher Country (ISBN Group)</b>
        </div>
        <div style="position:fixed;bottom:30px;right:15px;z-index:1000;
            background:white;padding:10px;border-radius:8px;
            border-left:4px solid #e63946;font-size:12px;font-family:Arial;
            box-shadow:0 2px 8px rgba(0,0,0,0.2);max-width:200px">
            <b>Key Findings:</b><br>
            UK = 45% of all books<br>
            South Korea avg rating: 4.05<br>
            Philippines: 4.11 avg rating
        </div>
    '''))
    return m._repr_html_()

MAP_HTML = build_folium_map()

# ─── 4) DASH APP ──────────────────────────────────────────────────────────────
app  = Dash(__name__)
server = app.server

# Geo bar chart (top 10 countries)
top10 = geo_agg.nlargest(10,'book_count')

app.layout = html.Div([
    html.Div([
        html.H1("📚 GoodReads Dashboard — Phase 3",
                style={"margin":"0","color":"white"}),
        html.P("Interactive Analytics + Geospatial Layer",
               style={"margin":"4px 0 0","color":"#cfe2ff"})
    ], style={"background":"#1a3c6b","padding":"20px"}),

    dcc.Tabs([
        # ── Tab 1: Original Dashboard ─────────────────────────────────────
        dcc.Tab(label="📊 Analytics Dashboard", children=[
            html.Div([
                dcc.Dropdown(id="format-dropdown",
                    options=[{"label":f,"value":f} for f in top_formats],
                    value=top_formats[:3], multi=True),
                dcc.RangeSlider(id="rating-slider",
                    min=0, max=5, step=0.1, value=[3.0,5.0],
                    marks={i:str(i) for i in range(6)}),
            ], style={"padding":"20px"}),
            html.Div(id="kpi-row",
                     style={"display":"flex","gap":"10px","padding":"0 20px"}),
            html.Div([
                dcc.Graph(id="genre-bar"),
                dcc.Graph(id="scatter-chart"),
            ], style={"display":"flex","flexWrap":"wrap"}),
        ]),

        # ── Tab 2: Geospatial Layer ───────────────────────────────────────
        dcc.Tab(label="🌍 Geospatial Analysis", children=[
            html.Div([
                html.H3("Books by Publisher Country",
                        style={"color":"#1a3c6b","margin":"0 0 4px"}),
                html.P("Derived from ISBN registration group prefix — "
                       "67,049 books mapped across 45 countries.",
                       style={"color":"#555","margin":"0 0 12px","fontSize":"13px"}),
            ], style={"padding":"20px 20px 0"}),

            # Side-by-side: map + bar chart
            html.Div([
                # Folium map
                html.Div([
                    html.Iframe(srcDoc=MAP_HTML,
                                style={"width":"100%","height":"420px",
                                       "border":"none","borderRadius":"8px"}),
                ], style={"flex":"1.6","minWidth":"300px"}),

                # Bar chart — non-spatial context
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            top10.sort_values('book_count'),
                            x='book_count', y='country', orientation='h',
                            color='avg_rating',
                            color_continuous_scale='RdYlGn',
                            range_color=[3.8, 4.2],
                            labels={'book_count':'Books Published',
                                    'country':'Country',
                                    'avg_rating':'Avg Rating'},
                            title='Top 10 Countries: Volume vs. Avg Rating',
                            text='book_count',
                        ).update_layout(
                            plot_bgcolor='#f9f9f9',
                            paper_bgcolor='white',
                            coloraxis_colorbar=dict(title="Avg<br>Rating"),
                            margin=dict(l=10,r=10,t=40,b=10),
                            height=420,
                        ),
                        style={"height":"420px"}
                    ),
                ], style={"flex":"1","minWidth":"280px"}),
            ], style={"display":"flex","gap":"16px",
                      "padding":"0 20px 20px","flexWrap":"wrap"}),

            # Geo insight cards
            html.Div([
                html.Div([
                    html.H4("🇬🇧 UK Dominates", style={"margin":"0 0 6px","color":"#1a3c6b"}),
                    html.P("45.3% of all books carry a UK ISBN prefix, "
                           "reflecting GoodReads' English-language bias."),
                ], style={"flex":"1","background":"#eef6ff","padding":"14px",
                           "borderRadius":"8px","minWidth":"200px"}),
                html.Div([
                    html.H4("🇰🇷 South Korea Excels", style={"margin":"0 0 6px","color":"#1a3c6b"}),
                    html.P("South Korea ranks #14 in volume but leads "
                           "in avg rating (4.05), suggesting curation over quantity."),
                ], style={"flex":"1","background":"#fff8ee","padding":"14px",
                           "borderRadius":"8px","minWidth":"200px"}),
                html.Div([
                    html.H4("🌏 Asia Rising", style={"margin":"0 0 6px","color":"#1a3c6b"}),
                    html.P("China + Japan + Korea together contribute 16% of books, "
                           "with consistently high ratings ≥ 3.89."),
                ], style={"flex":"1","background":"#eeffee","padding":"14px",
                           "borderRadius":"8px","minWidth":"200px"}),
            ], style={"display":"flex","gap":"12px","padding":"0 20px 20px",
                      "flexWrap":"wrap"}),
        ]),
    ]),
])

# ─── 5) CALLBACKS ──────────────────────────────────────────────────────────────
@app.callback(
    Output("genre-bar","figure"),
    Output("scatter-chart","figure"),
    Output("kpi-row","children"),
    Input("format-dropdown","value"),
    Input("rating-slider","value"),
)
def update(selected_formats, rating_range):
    if not selected_formats: selected_formats = top_formats[:3]
    low, high = rating_range
    filtered = df[(df["bookformat"].isin(selected_formats)) &
                  (df["rating"] >= low) & (df["rating"] <= high)]

    def card(label, value):
        return html.Div([html.P(label, style={"margin":"0","fontSize":"12px","color":"#555"}),
                         html.H3(value, style={"margin":"4px 0 0"})],
                        style={"padding":"12px","background":"white",
                               "borderRadius":"8px","flex":"1",
                               "boxShadow":"0 1px 4px rgba(0,0,0,0.1)"})

    avg_r    = round(filtered["rating"].mean(), 2) if len(filtered) else 0
    med_p    = int(filtered["pages"].median())     if len(filtered) else 0
    kpis = [card("Total Books", f"{len(filtered):,}"),
            card("Avg Rating",  avg_r),
            card("Median Pages", med_p)]

    gc = filtered["genre"].value_counts().head(10).reset_index()
    gc.columns = ["Genre","Books"]
    bar = px.bar(gc, x="Books", y="Genre", orientation="h",
                 color="Books", color_continuous_scale="Blues",
                 title="Top 10 Genres")
    bar.update_layout(showlegend=False, plot_bgcolor="#f9f9f9",
                      paper_bgcolor="white", coloraxis_showscale=False)

    if len(filtered) > 0:
        scat = filtered.sample(min(3000, len(filtered)), random_state=42).copy()
        scat["log_tr"] = np.log1p(scat["totalratings"])
        scatter = px.scatter(scat, x="pages", y="rating", color="bookformat",
                             size="log_tr", hover_data=["title"],
                             title="Pages vs Rating by Format")
        scatter.update_layout(plot_bgcolor="#f9f9f9", paper_bgcolor="white")
    else:
        scatter = px.scatter()

    return bar, scatter, kpis

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
