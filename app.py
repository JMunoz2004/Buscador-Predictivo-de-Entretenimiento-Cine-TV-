"""¿Qué Veo Hoy? — interfaz Streamlit.

Ejecutar en local:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from queveo import config as C  # noqa: E402
from queveo import tmdb_api  # noqa: E402
from queveo.data import load_catalog  # noqa: E402
from queveo.model import load_metrics, load_model  # noqa: E402
from queveo.recommender import Query, Recommender  # noqa: E402

st.set_page_config(page_title="¿Qué veo hoy?", page_icon="🍿", layout="wide")

# Clave de TMDb opcional (Streamlit Cloud -> Settings -> Secrets)
try:
    if "TMDB_API_KEY" in st.secrets:
        os.environ["TMDB_API_KEY"] = st.secrets["TMDB_API_KEY"]
except Exception:
    pass

# --------------------------------------------------------------------------- estilos
GENRE_TINT = {  # fondo de los pósters generados, por género principal
    "Comedy": "#F2C14E", "Animation": "#F28E4E", "Family": "#7FC8A9", "Action": "#B3122E",
    "Adventure": "#2E7D6B", "Horror": "#2B2240", "Thriller": "#3A3F63", "Mystery": "#4B3F72",
    "Drama": "#5C4B8A", "Romance": "#C2185B", "Science Fiction": "#1E4E79", "Fantasy": "#6A3FA0",
    "Crime": "#41394F", "Documentary": "#5F7161", "History": "#8C6A3F", "War": "#5A5A3C",
    "Music": "#C25E8C", "Western": "#A0522D",
}
LIGHT_TINTS = {"#F2C14E", "#F28E4E", "#7FC8A9"}

st.html(
    """
<style>
/* Fuentes servidas desde static/ (sin depender de Google Fonts) */
@font-face { font-family:'Big Shoulders Display'; font-weight:700; font-style:normal; font-display:swap; src:url('/app/static/fonts/big-shoulders-display-latin-700-normal.woff2') format('woff2'); }
@font-face { font-family:'Big Shoulders Display'; font-weight:800; font-style:normal; font-display:swap; src:url('/app/static/fonts/big-shoulders-display-latin-800-normal.woff2') format('woff2'); }
@font-face { font-family:'Big Shoulders Display'; font-weight:900; font-style:normal; font-display:swap; src:url('/app/static/fonts/big-shoulders-display-latin-900-normal.woff2') format('woff2'); }
@font-face { font-family:'Atkinson Hyperlegible'; font-weight:400; font-style:normal; font-display:swap; src:url('/app/static/fonts/atkinson-hyperlegible-latin-400-normal.woff2') format('woff2'); }
@font-face { font-family:'Atkinson Hyperlegible'; font-weight:700; font-style:normal; font-display:swap; src:url('/app/static/fonts/atkinson-hyperlegible-latin-700-normal.woff2') format('woff2'); }
:root { --ink:#1F1A3D; --room:#ECEAF4; --seat:#B3122E; --corn:#F2C14E; --muted:#6B6785; --paper:#FFFFFF; }
html, body, [class*="css"], .stApp { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--room); }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1, h2, h3, .marquee { font-family: 'Big Shoulders Display', 'Arial Narrow', sans-serif !important; letter-spacing: .01em; color: var(--ink); }
.hero { display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.hero h1 { font-size: clamp(3rem, 7vw, 5.6rem); line-height:.92; font-weight:900; margin:0; padding:0; }
.hero p { max-width: 34ch; margin:0 0 .5rem; color: var(--muted); font-size:1.05rem; line-height:1.45; }

/* Entradas */
.ticket { display:grid; grid-template-columns: 120px 1fr auto; background:var(--paper); border-radius:14px;
          position:relative; overflow:hidden; border:1px solid #D9D6E8; }
.ticket.big { grid-template-columns: 200px 1fr 190px; }
.poster { aspect-ratio: 2/3; width:100%; display:flex; align-items:flex-end; padding:.6rem; box-sizing:border-box;
          font-family:'Big Shoulders Display','Arial Narrow',sans-serif; font-weight:800; font-size:1.05rem; line-height:1; color:#fff;
          background-size:cover; background-position:center; }
.poster.light { color: var(--ink); }
.ticket.big .poster { font-size:1.6rem; padding:1rem; }
.body { padding: .9rem 1.1rem; min-width:0; }
.body h3 { font-size:1.55rem; font-weight:800; margin:0 0 .15rem; line-height:1; padding:0; }
.ticket.big .body h3 { font-size:2.6rem; }
.meta { color:var(--muted); font-size:.88rem; margin-bottom:.5rem; }
.reasons { margin:.4rem 0 0 !important; padding:0 !important; list-style:none; font-size:.92rem; }
.reasons li { padding-left:1.1rem; position:relative; margin-bottom:.15rem; }
.reasons li::before { content:"✓"; position:absolute; left:0; color:var(--seat); font-weight:700; }
.overview { font-size:.95rem; line-height:1.5; max-width:62ch; margin:.4rem 0 .6rem; }
.chips span { display:inline-block; white-space:nowrap; font-size:.78rem; padding:.12rem .5rem; border-radius:999px; margin:0 .25rem .25rem 0;
              background:#F1EFF8; border:1px solid #DCD8EC; }
.chips span.pf { background: var(--ink); color:#fff; border-color: var(--ink); }
/* Talón de la entrada con el match */
.stub { border-left: 2px dashed #CFCBE2; padding: .9rem 1rem; display:flex; flex-direction:column;
        align-items:center; justify-content:center; text-align:center; min-width:92px; }
.stub .n { font-family:'Big Shoulders Display','Arial Narrow',sans-serif; font-weight:900; font-size:2.6rem; line-height:.9; color:var(--ink); }
.ticket.big .stub { background: var(--corn); border-left-color: var(--ink); }
.ticket.big .stub .n { font-size:5.4rem; }
.stub .l { font-size:.8rem; color:var(--muted); }
.ticket.big .stub .l { color: var(--ink); }
.ticket::before, .ticket::after { content:""; position:absolute; width:22px; height:22px; border-radius:50%;
        background:var(--room); right:calc(var(--stubw, 92px) + 1.9rem - 11px); }
.ticket::before { top:-11px; } .ticket::after { bottom:-11px; }
.ticket.big { --stubw: 190px; }
.ticket.big::before, .ticket.big::after { right: calc(190px - 11px); }
.ticket.small::before, .ticket.small::after { display:none; }
.ticket.small { grid-template-columns: 84px 1fr; }
.ticket.small .poster { font-size:.9rem; padding:.45rem; }
.ticket.small .body { padding:.7rem .85rem; }
.ticket.small .body h3 { font-size:1.35rem; }
.ticket.small .meta { font-size:.8rem; margin-bottom:.35rem; }
.ticket.small .reasons { font-size:.84rem; }
.ticket.small .stub { grid-column: 1 / -1; flex-direction:row; justify-content:flex-start; align-items:baseline; gap:.5rem;
        border-left:none; border-top:2px dashed #CFCBE2; padding:.45rem .85rem; }
.ticket.small .stub .n { font-size:1.9rem; }
.stButtonGroup div:has(> button) { flex-wrap: wrap !important; overflow: visible !important; row-gap:.4rem; }
.stApp button, .stApp input, .stApp label, .stApp p { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
.note { font-size:.85rem; color:var(--muted); }
div.stButton > button[kind="primary"] { background: var(--seat); border-color: var(--seat); font-weight:700;
        font-size:1.05rem; padding:.55rem 1.6rem; border-radius:999px; }
div.stButton > button[kind="primary"]:hover { background:#8E0E24; border-color:#8E0E24; }
div.stButton > button:focus-visible { outline: 3px solid var(--ink); outline-offset:2px; }
@media (max-width: 760px) {
  .ticket.big, .ticket { grid-template-columns: 96px 1fr; }
  .ticket .stub, .ticket.big .stub { grid-column: 1 / -1; border-left:none; border-top:2px dashed #CFCBE2;
        flex-direction:row; gap:.6rem; padding:.5rem; }
  .ticket.big .stub .n { font-size:3rem; }
  .ticket.big .body h3 { font-size:1.9rem; }
  .ticket::before, .ticket::after { display:none; }
}
</style>
"""
)


# --------------------------------------------------------------------------- recursos
@st.cache_resource(show_spinner="Preparando el catálogo…")
def get_recommender() -> Recommender:
    try:
        model = load_model()
    except Exception:  # p. ej. versión de scikit-learn distinta a la del entrenamiento
        from queveo.data import load_ratings
        from queveo.model import save, train_and_evaluate
        res = train_and_evaluate(load_ratings(), load_catalog())
        save(res)
        model = res.model
    return Recommender(load_catalog(), model, load_metrics())


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_info(tmdb_id: int) -> dict:
    return tmdb_api.movie_details(tmdb_id)


rec = get_recommender()
cat = rec.cat


def miles(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def genres_es(gs) -> list[str]:
    return [C.GENRES_ES.get(g, g) for g in gs]


def poster_html(r: pd.Series, info: dict) -> str:
    tint = GENRE_TINT.get(r["main_genre"], "#5C4B8A")
    if info.get("poster"):
        return f'<div class="poster" role="img" aria-label="Póster de {html.escape(r["title"])}" ' \
               f'style="background-image:url({info["poster"]})"></div>'
    cls = "poster light" if tint in LIGHT_TINTS else "poster"
    genre = C.GENRES_ES.get(r["main_genre"], "")
    return f'<div class="{cls}" style="background:{tint}" aria-hidden="true">{genre}</div>'


def ticket_html(r: pd.Series, big: bool = False) -> str:
    info = tmdb_info(int(r["tmdb_id"])) if tmdb_api.enabled() else {}
    title = info.get("title_es") or r["title"]
    platforms = info.get("providers_es") if info.get("providers_es") is not None else r["platforms"]
    meta = f'{r["year"]}, {r["runtime"]} min, {", ".join(genres_es(r["genres"][:3])).lower()}'
    if r["director"]:
        meta += f'. Dirige {html.escape(r["director"])}'
    reasons = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
    chips = "".join(f'<span class="pf">{p}</span>' for p in platforms) or "<span>No disponible en streaming</span>"
    overview = ""
    if big and r["overview"]:
        ov = r["overview"] if len(r["overview"]) < 320 else r["overview"][:317].rsplit(" ", 1)[0] + "…"
        overview = f'<p class="overview">{html.escape(ov)}</p>'
    out = f"""
<div class="ticket {'big' if big else 'small'}">
  {poster_html(r, info)}
  <div class="body">
    <h3>{html.escape(title)}</h3>
    <div class="meta">{meta}</div>
    {overview}
    <div class="chips">{chips}</div>
    <ul class="reasons">{reasons if big else reasons.split('</li>')[0] + '</li>' if reasons else ''}</ul>
  </div>
  <div class="stub"><div class="n">{r['match']:.0f}%</div><div class="l">de match</div></div>
</div>"""
    # sin sangría: Streamlit interpretaría las líneas con 4 espacios como bloque de código
    return "".join(line.strip() for line in out.splitlines())


# --------------------------------------------------------------------------- cabecera
st.markdown(
    """<div class="hero"><h1>¿Qué veo<br>hoy?</h1>
    <p>Dinos cuánto tiempo tienes y cómo te sientes. Te proponemos la película que más probablemente vas a disfrutar esta noche.</p></div>""",
    unsafe_allow_html=True,
)

tab_rec, tab_how = st.tabs(["Elegir película", "Cómo funciona"])

# --------------------------------------------------------------------------- pestaña principal
with tab_rec:
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        mood = st.pills("¿Cómo te sientes?", list(C.MOODS), default="😂 Quiero reírme", selection_mode="single")
        minutes = st.slider("¿Cuánto tiempo tienes?", 60, 240, 120, step=5, format="%d min",
                            help="Solo te proponemos películas que quepan en ese tiempo (con 5 min de margen).")
    with c2:
        platforms = st.pills("Tus plataformas", C.PLATFORMS, default=C.PLATFORMS, selection_mode="multi")
        era = st.pills("Época", list(C.ERAS), default="Me da igual", selection_mode="single")

    with st.expander("Afinar más: películas que te encantan y nota mínima"):
        options = cat.sort_values("vote_count", ascending=False)
        labels = dict(zip(options["tmdb_id"], options["title"] + " (" + options["year"].astype(str) + ")"))
        liked = st.multiselect("Películas que te encantan", list(labels), format_func=labels.get, max_selections=5,
                               placeholder="Escribe un título, por ejemplo Inception",
                               help="Ajustan tu perfil de gustos y buscamos películas con temas parecidos.")
        min_rating = st.slider("Nota mínima en TMDb", 0.0, 8.0, 6.0, step=0.5, format="%.1f")

    if "seed" not in st.session_state:
        st.session_state.seed = 0
    b1, b2, _ = st.columns([1.2, 1.2, 3])
    if b1.button("Buscar mi película", type="primary", width="stretch"):
        st.session_state.seed = 0
    if b2.button("Otra tanda", width="stretch",
                 help="En modo Sorpréndeme cambia la selección; en el resto muestra las siguientes opciones."):
        st.session_state.seed += 1

    q = Query(minutes=minutes, mood=mood or "🎲 Sorpréndeme", era=era or "Me da igual",
              platforms=list(platforms or []), liked_ids=list(liked), min_rating=min_rating,
              top_n=10 + (0 if (mood == "🎲 Sorpréndeme") else 9 * st.session_state.seed),
              seed=st.session_state.seed)
    results = rec.recommend(q)
    page = 9 * st.session_state.seed
    if mood != "🎲 Sorpréndeme" and page:
        results = results.iloc[page:] if len(results) > page else results

    st.write("")
    if not platforms:
        st.info("Marca al menos una plataforma para ver propuestas.")
    elif results.empty:
        st.warning("Ninguna película cumple todo a la vez. Prueba con más tiempo, otra época o una nota mínima más baja.")
    else:
        top = results.iloc[0]
        st.markdown(ticket_html(top, big=True), unsafe_allow_html=True)

        with st.expander(f"Más sobre {top['title']}"):
            d1, d2 = st.columns([1, 1.3], gap="large")
            with d1:
                comp = {"Satisfacción prevista": top["s_satisfaccion"], "Encaje con tu ánimo": top["s_animo"],
                        "Temas parecidos": top["s_contenido"], "Aprovecha tu tiempo": top["s_tiempo"]}
                comp = {k: v for k, v in comp.items() if pd.notna(v)}
                fig = go.Figure(go.Bar(x=[v * 100 for v in comp.values()], y=list(comp), orientation="h",
                                       marker_color=["#1F1A3D", "#B3122E", "#5C4B8A", "#F2C14E"][:len(comp)],
                                       text=[f"{v * 100:.0f}" for v in comp.values()], textposition="outside"))
                fig.update_layout(height=210, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 110], visible=False),
                                  yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Atkinson Hyperlegible", color="#1F1A3D"))
                st.markdown("**De dónde sale su match**")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(f"Nota prevista para ti: {top['pred_rating']:.1f} de 5. "
                           f"En TMDb: {top['vote_average']:.1f} de 10 con {miles(top['vote_count'])} votos.")
            with d2:
                if len(top["cast"]) > 0:
                    st.markdown(f"**Reparto:** {', '.join(top['cast'])}")
                st.markdown("**Si te gusta, también te puede gustar**")
                sim = rec.similar(int(top["tmdb_id"]), k=5, exclude=list(liked))
                for _, s in sim.iterrows():
                    st.markdown(f"- {s['title']} ({s['year']}), {s['runtime']} min")

        st.markdown('<h2 style="font-size:2rem;margin:1.6rem 0 .6rem">Otras buenas opciones</h2>', unsafe_allow_html=True)
        rest = results.iloc[1:10]
        cols = st.columns(3, gap="medium")
        for i, (_, r) in enumerate(rest.iterrows()):
            with cols[i % 3]:
                st.markdown(ticket_html(r), unsafe_allow_html=True)
                st.write("")

    if not tmdb_api.enabled():
        st.markdown('<p class="note">La disponibilidad por plataforma es una aproximación a partir de la productora '
                    'de cada película. Ver «Cómo funciona».</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- cómo funciona
with tab_how:
    m = load_metrics()
    st.markdown("## Cómo elegimos tu película")
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown(f"""
1. **Datos.** {miles(len(cat))} películas de TMDb (sinopsis, géneros, duración, reparto) unidas con
   {miles(m['n_train'] + m['n_test'])} valoraciones reales de {m['n_users']} usuarios de MovieLens.
2. **Tu perfil.** Tu estado de ánimo, la época y las películas que te encantan se traducen en un perfil
   de gustos con la misma forma que los perfiles de los usuarios reales: cuánto te gusta cada género y
   qué época prefieres.
3. **Predicción.** Un modelo de *gradient boosting* entrenado con esos usuarios predice la nota que le
   darías a cada película del catálogo.
4. **Match.** Combinamos la nota prevista (45 %), el encaje con tu ánimo (25 %), la similitud de temas con
   lo que buscas, medida con TF-IDF sobre sinopsis y palabras clave (20 %), y lo bien que aprovecha tu
   tiempo (10 %).
""")
    with b:
        t = pd.DataFrame(m["modelos"]).T[["RMSE", "MAE", "precision@5"]]
        t.columns = ["RMSE", "MAE", "Precisión@5"]
        st.markdown("**Evaluación en valoraciones no vistas (20 % por usuario)**")
        st.dataframe(t.style.format("{:.3f}").highlight_min(subset=["RMSE", "MAE"], color="#F2C14E")
                     .highlight_max(subset=["Precisión@5"], color="#F2C14E"), width="stretch")
        st.caption("El modelo híbrido se equivoca de media 0,64 estrellas, frente a 0,69 del mejor método "
                   "clásico. Precisión@5: proporción de las 5 primeras propuestas que el usuario puntuó con 4 o más.")

    imp = pd.Series(m["importancia_top"]).head(8).iloc[::-1]
    nice = {"vote_average": "Nota media en TMDb", "user_mean": "Exigencia del usuario", "aff_dot": "Afinidad con sus géneros",
            "year": "Año de estreno", "log_budget": "Presupuesto", "log_votes": "Nº de votos", "aff_min": "Género menos afín",
            "user_pref_year": "Época preferida", "year_gap": "Distancia a su época", "aff_max": "Género más afín",
            "runtime": "Duración", "log_popularity": "Popularidad"}
    fig = go.Figure(go.Bar(x=imp.values, y=[nice.get(k, k) for k in imp.index], orientation="h", marker_color="#5C4B8A"))
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=30, b=0), title="Qué pesa más en la predicción",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Atkinson Hyperlegible", color="#1F1A3D"), xaxis_title="Aumento del error al quitarla")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("""
**Limitaciones conocidas**

- El catálogo de TMDb 5000 llega hasta 2016 e incluye solo películas; las series son el siguiente paso.
- TMDb 5000 no trae disponibilidad en streaming. Sin clave de API, se asigna a partir de la productora
  (Disney y Pixar en Disney+, Warner en HBO Max, MGM en Prime Video…) más una asignación estable que imita
  las licencias. Con `TMDB_API_KEY` configurada, la app muestra pósters y la disponibilidad real en España.
""")
