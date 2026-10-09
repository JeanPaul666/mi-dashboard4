"""Dashboard de ventas para una cadena de restaurantes (Streamlit + Plotly).

Datos esperados (CSV o Excel) con estas columnas:
FECHA, HORA, LOCAL, ARTICULO, TIPO_ARTICULO, CANTIDAD, PRECIO_UNITARIO, TOTAL
(DISTRITO es opcional: se toma de la tabla de locales de abajo).

El archivo se busca en  Origen/articles.csv  o  articles.csv  junto a este app.py.
También se puede subir desde la barra lateral. Si no hay datos, se muestran datos demo.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard de Ventas", page_icon="🍽️", layout="wide")

# ----------------------------------------------------------------------
# Configuración: nombres de columnas (los mismos del notebook)
# ----------------------------------------------------------------------
ID_FECHA = "FECHA"
ID_HORA = "HORA"
ID_LOCAL = "LOCAL"
ID_DISTRITO = "DISTRITO"
ID_ZONA = "ZONA"
ID_ARTICULO = "ARTICULO"
ID_TIPO_ARTICULO = "TIPO_ARTICULO"
ID_CANTIDAD = "CANTIDAD"
ID_TOTAL = "TOTAL"
ID_PRECIO_UNITARIO = "PRECIO_UNITARIO"

RUTAS_DATOS = ["Origen/articles.csv", "articles.csv"]

# Los 10 locales de muestra (coordenadas aproximadas del centro de cada distrito)
LOCALES = pd.DataFrame(
    [
        (3, "Breña", "CENTRO", -12.0570, -77.0530),
        (5, "Rímac", "NORTE", -12.0290, -77.0430),
        (9, "San Juan de Lurigancho", "ESTE", -11.9700, -77.0050),
        (12, "San Juan de Miraflores", "SUR", -12.1560, -76.9740),
        (14, "Santa Anita", "ESTE", -12.0450, -76.9720),
        (22, "Los Olivos", "NORTE", -11.9700, -77.0720),
        (24, "San Miguel", "CENTRO", -12.0770, -77.0830),
        (25, "Miraflores", "SUR", -12.1190, -77.0290),
        (26, "San Borja", "SUR", -12.1000, -76.9990),
        (27, "Barranco", "SUR", -12.1400, -77.0210),
    ],
    columns=[ID_LOCAL, ID_DISTRITO, ID_ZONA, "LAT", "LON"],
)
ETIQUETA_LOCAL = {
    int(l): f"{l} · {d} ({z.title()})"
    for l, d, z in zip(LOCALES[ID_LOCAL], LOCALES[ID_DISTRITO], LOCALES[ID_ZONA])
}
ZONAS = ["CENTRO", "NORTE", "ESTE", "SUR"]
COLOR_ZONA = {"CENTRO": "#E4572E", "NORTE": "#29335C", "ESTE": "#F3A712", "SUR": "#2E9CCA"}
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Carga y preparación de datos
# ----------------------------------------------------------------------
def leer_tabla(origen, nombre):
    """Lee CSV o Excel; prueba utf-8 / latin-1 y separador , o ;"""
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(origen)
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(origen, "seek"):
                origen.seek(0)
            df = pd.read_csv(origen, encoding=enc)
            if df.shape[1] == 1:
                if hasattr(origen, "seek"):
                    origen.seek(0)
                df = pd.read_csv(origen, encoding=enc, sep=";")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("No pude leer el archivo (revisa que sea CSV o Excel).")


def preparar(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.upper()

    if ID_TOTAL not in df.columns and {ID_CANTIDAD, ID_PRECIO_UNITARIO} <= set(df.columns):
        df[ID_TOTAL] = (
            pd.to_numeric(df[ID_CANTIDAD], errors="coerce")
            * pd.to_numeric(df[ID_PRECIO_UNITARIO], errors="coerce")
        )

    requeridas = [ID_FECHA, ID_HORA, ID_LOCAL, ID_ARTICULO, ID_TIPO_ARTICULO, ID_CANTIDAD, ID_TOTAL]
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise KeyError(
            f"Faltan columnas: {', '.join(faltan)}. Columnas encontradas: {', '.join(df.columns)}"
        )

    # Fecha
    fechas = pd.to_datetime(df[ID_FECHA], errors="coerce")
    if fechas.isna().mean() > 0.5:
        fechas = pd.to_datetime(df[ID_FECHA], errors="coerce", dayfirst=True)
    df[ID_FECHA] = fechas
    df = df.dropna(subset=[ID_FECHA])

    # Hora (acepta 13, "13:45:00" o 1345)
    hora_num = pd.to_numeric(df[ID_HORA], errors="coerce")
    if hora_num.notna().mean() > 0.9:
        hora = hora_num // 100 if hora_num.max() > 24 else hora_num
    else:
        hora = pd.to_numeric(df[ID_HORA].astype(str).str.extract(r"(\d{1,2})\s*:")[0], errors="coerce")
    df["HORA_N"] = hora.astype(float).round()

    # Local -> zona, distrito y coordenadas (solo los 10 locales de muestra)
    df[ID_LOCAL] = pd.to_numeric(df[ID_LOCAL].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.dropna(subset=[ID_LOCAL])
    df[ID_LOCAL] = df[ID_LOCAL].astype(int)
    df = df.merge(LOCALES, on=ID_LOCAL, how="inner", suffixes=("_ORIG", ""))

    for c in (ID_CANTIDAD, ID_TOTAL):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[ID_ARTICULO] = df[ID_ARTICULO].astype(str).str.strip()
    df[ID_TIPO_ARTICULO] = df[ID_TIPO_ARTICULO].astype(str).str.strip()

    # Columnas derivadas
    df["MES"] = df[ID_FECHA].dt.to_period("M").astype(str)
    df["DIA"] = df[ID_FECHA].dt.normalize()
    df["DIA_SEMANA"] = pd.Categorical(
        df[ID_FECHA].dt.dayofweek.map(dict(enumerate(DIAS))), categories=DIAS, ordered=True
    )
    return df


@st.cache_data(show_spinner="Cargando datos...")
def cargar_desde_ruta(ruta):
    return preparar(leer_tabla(ruta, ruta))


@st.cache_data(show_spinner="Procesando archivo...")
def cargar_desde_bytes(contenido, nombre):
    return preparar(leer_tabla(io.BytesIO(contenido), nombre))


@st.cache_data(show_spinner="Generando datos de demostración...")
def cargar_demo():
    rng = np.random.default_rng(7)
    catalogo = {
        "Pollo a la brasa": [("1/4 Pollo a la brasa", 24), ("1/2 Pollo a la brasa", 42), ("Pollo entero", 78)],
        "Parrillas": [("Anticuchos", 28), ("Parrilla personal", 55), ("Churrasco", 38)],
        "Acompañamientos": [("Papas fritas", 12), ("Ensalada", 10), ("Choclo con queso", 14)],
        "Bebidas": [("Chicha morada", 9), ("Inca Kola 500ml", 6), ("Cerveza", 12), ("Limonada", 10)],
        "Postres": [("Suspiro limeño", 14), ("Cheesecake", 16), ("Helado", 10)],
    }
    items = [(fam, art, precio) for fam, lista in catalogo.items() for art, precio in lista]
    familias = np.array([i[0] for i in items])
    articulos = np.array([i[1] for i in items])
    precios = np.array([i[2] for i in items], dtype=float)
    peso_art = rng.uniform(0.5, 2.0, len(items))
    peso_art /= peso_art.sum()

    horas = np.arange(11, 23)
    peso_hora = np.array([2, 6, 9, 8, 4, 2, 2, 4, 7, 9, 6, 2], dtype=float)
    peso_hora /= peso_hora.sum()

    base_local = {int(l): rng.uniform(0.7, 1.4) for l in LOCALES[ID_LOCAL]}
    bloques = []
    for dia in pd.date_range("2026-09-01", "2026-09-30"):
        factor = 1.6 if dia.dayofweek >= 4 else 1.0
        for local, base in base_local.items():
            n = int(rng.poisson(110 * base * factor))
            idx = rng.choice(len(items), n, p=peso_art)
            cant = rng.integers(1, 5, n)
            bloques.append(
                pd.DataFrame(
                    {
                        ID_FECHA: dia,
                        ID_HORA: rng.choice(horas, n, p=peso_hora),
                        ID_LOCAL: local,
                        ID_ARTICULO: articulos[idx],
                        ID_TIPO_ARTICULO: familias[idx],
                        ID_CANTIDAD: cant,
                        ID_PRECIO_UNITARIO: precios[idx],
                        ID_TOTAL: cant * precios[idx],
                    }
                )
            )
    return preparar(pd.concat(bloques, ignore_index=True))


def soles(x):
    return f"S/ {x:,.0f}"


def mostrar(fig, alto=380):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------
# Barra lateral: origen de datos y filtros
# ----------------------------------------------------------------------
st.sidebar.title("🍽️ Filtros")
archivo = st.sidebar.file_uploader("Subir datos (CSV o Excel)", type=["csv", "xlsx"])

es_demo = False
try:
    if archivo is not None:
        df = cargar_desde_bytes(archivo.getvalue(), archivo.name)
        fuente = f"Archivo subido: {archivo.name}"
    else:
        bases = [Path(__file__).resolve().parent, Path.cwd()]
        ruta = next((b / r for b in bases for r in RUTAS_DATOS if (b / r).exists()), None)
        if ruta is not None:
            df = cargar_desde_ruta(str(ruta))
            fuente = f"Archivo: {ruta.name}"
        else:
            df = cargar_demo()
            fuente = "Datos de demostración"
            es_demo = True
except Exception as e:  # noqa: BLE001
    st.error(f"No pude cargar los datos: {e}")
    st.stop()

if df.empty:
    st.error(
        "Ninguna fila coincide con los 10 locales de muestra "
        f"({', '.join(str(x) for x in LOCALES[ID_LOCAL])}). Revisa la columna LOCAL."
    )
    st.stop()

meses = sorted(df["MES"].unique())
mes = st.sidebar.selectbox("Mes de análisis", meses, index=len(meses) - 1)
df_mes = df[df["MES"] == mes]

zonas = st.sidebar.multiselect("Zona", ZONAS, default=ZONAS)
locales_disp = [int(x) for x in LOCALES.loc[LOCALES[ID_ZONA].isin(zonas), ID_LOCAL]]
locales_sel = st.sidebar.multiselect(
    "Local", locales_disp, default=locales_disp, format_func=lambda x: ETIQUETA_LOCAL[x]
)

todas_familias = sorted(df_mes[ID_TIPO_ARTICULO].unique())
familias = st.sidebar.multiselect("Familia de producto", todas_familias, default=todas_familias)

dmin, dmax = df_mes[ID_FECHA].min().date(), df_mes[ID_FECHA].max().date()
rango = st.sidebar.date_input("Rango de fechas", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rango, (tuple, list)) and len(rango) == 2:
    ini, fin = rango
elif isinstance(rango, (tuple, list)) and len(rango) == 1:
    ini = fin = rango[0]
else:
    ini = fin = rango

dias_sel = st.sidebar.multiselect("Días de la semana", DIAS, default=DIAS)

horas_validas = df_mes["HORA_N"].dropna()
if horas_validas.empty:
    h_ini, h_fin = 0, 23
else:
    hmin, hmax = int(horas_validas.min()), int(horas_validas.max())
    if hmin < hmax:
        h_ini, h_fin = st.sidebar.slider("Rango de horas", hmin, hmax, (hmin, hmax))
    else:
        h_ini, h_fin = hmin, hmax

st.sidebar.caption(f"📂 {fuente}")

f = df_mes[
    df_mes[ID_ZONA].isin(zonas)
    & df_mes[ID_LOCAL].isin(locales_sel)
    & df_mes[ID_TIPO_ARTICULO].isin(familias)
    & df_mes["DIA"].between(pd.Timestamp(ini), pd.Timestamp(fin))
    & df_mes["DIA_SEMANA"].isin(dias_sel)
    & (df_mes["HORA_N"].between(h_ini, h_fin) | df_mes["HORA_N"].isna())
]

# ----------------------------------------------------------------------
# Encabezado y KPIs
# ----------------------------------------------------------------------
st.title(f"🍽️ Dashboard de Ventas · {mes}")
if es_demo:
    st.info(
        "Estás viendo **datos de demostración**. Para usar los tuyos, guarda el archivo en "
        "`Origen/articles.csv` (junto a app.py) o súbelo en la barra lateral."
    )

if f.empty:
    st.warning("No hay ventas con esos filtros. Prueba ampliando zona, fechas u horas.")
    st.stop()

ventas = f[ID_TOTAL].sum()
unidades = f[ID_CANTIDAD].sum()
n_trans = len(f)
por_local = f.groupby(ID_LOCAL)[ID_TOTAL].sum()
local_top = int(por_local.idxmax())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ventas totales", soles(ventas))
c2.metric("Unidades vendidas", f"{unidades:,.0f}")
c3.metric("Ticket promedio", f"S/ {ventas / n_trans:,.2f}")
c4.metric("Transacciones", f"{n_trans:,}")
c5.metric("Local líder", f"Local {local_top}", ETIQUETA_LOCAL[local_top].split("· ")[1])

por_hora = f.dropna(subset=["HORA_N"]).groupby("HORA_N")[ID_TOTAL].sum()
hora_pico = f"{int(por_hora.idxmax()):02d}:00" if not por_hora.empty else "—"
por_dia_sem = f.groupby("DIA_SEMANA", observed=True)[ID_TOTAL].sum()
dia_pico = str(por_dia_sem.idxmax()) if not por_dia_sem.empty else "—"
st.caption(f"⏰ Hora pico: **{hora_pico}**  ·  📅 Día más fuerte: **{dia_pico}**")

tab_res, tab_tmp, tab_geo, tab_prod, tab_datos = st.tabs(
    ["📈 Resumen", "🕒 Temporal", "🗺️ Geográfico", "🍗 Productos", "📋 Datos"]
)

# ----------------------------- Resumen --------------------------------
with tab_res:
    diario = f.groupby("DIA")[ID_TOTAL].sum().reset_index()
    fig = px.bar(
        diario, x="DIA", y=ID_TOTAL, title="Ventas por día",
        labels={"DIA": "Fecha", ID_TOTAL: "Ventas (S/)"}, color_discrete_sequence=["#2E9CCA"],
    )
    fig.add_scatter(
        x=diario["DIA"], y=diario[ID_TOTAL].rolling(7, min_periods=1).mean(),
        mode="lines", name="Promedio 7 días", line=dict(color="#E4572E", width=3),
    )
    mostrar(fig)

    izq, der = st.columns(2)
    with izq:
        fam = f.groupby(ID_TIPO_ARTICULO)[ID_TOTAL].sum().reset_index()
        fig = px.pie(fam, names=ID_TIPO_ARTICULO, values=ID_TOTAL, hole=0.5,
                     title="Participación por familia")
        mostrar(fig)
    with der:
        rank = por_local.reset_index()
        rank["LOCAL_TXT"] = rank[ID_LOCAL].map(ETIQUETA_LOCAL)
        rank = rank.sort_values(ID_TOTAL)
        fig = px.bar(rank, x=ID_TOTAL, y="LOCAL_TXT", orientation="h", title="Ranking de locales",
                     labels={ID_TOTAL: "Ventas (S/)", "LOCAL_TXT": ""}, color_discrete_sequence=["#29335C"])
        mostrar(fig)

# ----------------------------- Temporal -------------------------------
with tab_tmp:
    st.subheader("¿Cuándo se vende más? Día de la semana × hora")
    h = f.dropna(subset=["HORA_N"]).copy()
    if h.empty:
        st.info("No se pudo leer la hora en los datos.")
    else:
        h["HORA_N"] = h["HORA_N"].astype(int)
        piv = h.pivot_table(index="DIA_SEMANA", columns="HORA_N", values=ID_TOTAL,
                            aggfunc="sum", observed=False)
        piv.index = piv.index.astype(str)
        piv = piv.reindex(DIAS).fillna(0)
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(x="Hora del día", y="", color="Ventas (S/)"))
        fig.update_xaxes(dtick=1)
        mostrar(fig, 420)

    izq, der = st.columns(2)
    with izq:
        sem = f.groupby("DIA_SEMANA", observed=False)[ID_TOTAL].sum().reindex(DIAS).reset_index()
        fig = px.bar(sem, x="DIA_SEMANA", y=ID_TOTAL, title="Ventas por día de la semana",
                     labels={"DIA_SEMANA": "", ID_TOTAL: "Ventas (S/)"},
                     color_discrete_sequence=["#F3A712"])
        mostrar(fig)
    with der:
        hh = por_hora.reset_index()
        fig = px.line(hh, x="HORA_N", y=ID_TOTAL, markers=True, title="Ventas por hora",
                      labels={"HORA_N": "Hora", ID_TOTAL: "Ventas (S/)"},
                      color_discrete_sequence=["#E4572E"])
        fig.update_xaxes(dtick=1)
        mostrar(fig)

# ----------------------------- Geográfico -----------------------------
with tab_geo:
    ventas_loc = (
        f.groupby(ID_LOCAL)
        .agg(VENTAS=(ID_TOTAL, "sum"), UNIDADES=(ID_CANTIDAD, "sum"),
             TRANSACCIONES=(ID_TOTAL, "count"), TICKET=(ID_TOTAL, "mean"))
        .reset_index()
    )
    mapa = LOCALES[LOCALES[ID_LOCAL].isin(locales_sel)].merge(ventas_loc, on=ID_LOCAL, how="left")
    mapa = mapa.fillna({"VENTAS": 0, "UNIDADES": 0, "TRANSACCIONES": 0, "TICKET": 0})
    mapa["ETIQUETA"] = mapa[ID_LOCAL].map(ETIQUETA_LOCAL)
    mapa["TAMANO"] = mapa["VENTAS"].clip(lower=1)

    spread = max(mapa["LAT"].max() - mapa["LAT"].min(), mapa["LON"].max() - mapa["LON"].min())
    zoom = 10.0 if spread > 0.15 else 10.8 if spread > 0.08 else 11.8 if spread > 0.03 else 12.8
    opciones = dict(
        lat="LAT", lon="LON", size="TAMANO", color=ID_ZONA, color_discrete_map=COLOR_ZONA,
        hover_name="ETIQUETA", size_max=45, zoom=zoom,
        center=dict(lat=float(mapa["LAT"].mean()), lon=float(mapa["LON"].mean())),
        hover_data={"VENTAS": ":,.0f", "UNIDADES": ":,.0f", "TICKET": ":,.2f",
                    "LAT": False, "LON": False, "TAMANO": False, ID_ZONA: False},
    )
    st.subheader("Mapa de ventas por local")
    st.caption("El tamaño del círculo es proporcional a las ventas. Cambia la zona en la barra lateral y el mapa se reenfoca.")
    try:
        fig = px.scatter_map(mapa, map_style="open-street-map", **opciones)
    except AttributeError:  # versiones antiguas de plotly
        fig = px.scatter_mapbox(mapa, mapbox_style="open-street-map", **opciones)
    mostrar(fig, 520)

    izq, der = st.columns(2)
    with izq:
        zona_df = f.groupby(ID_ZONA)[ID_TOTAL].sum().reset_index()
        fig = px.bar(zona_df, x=ID_ZONA, y=ID_TOTAL, color=ID_ZONA, color_discrete_map=COLOR_ZONA,
                     text_auto=",.0f", title="Ventas por zona", labels={ID_TOTAL: "Ventas (S/)"})
        fig.update_layout(showlegend=False)
        mostrar(fig)
    with der:
        tabla = mapa.sort_values("VENTAS", ascending=False)[
            [ID_LOCAL, ID_DISTRITO, ID_ZONA, "VENTAS", "UNIDADES", "TRANSACCIONES", "TICKET"]
        ]
        st.markdown("**Detalle por local**")
        st.dataframe(
            tabla, hide_index=True, width="stretch",
            column_config={
                "VENTAS": st.column_config.NumberColumn("Ventas (S/)", format="%.0f"),
                "UNIDADES": st.column_config.NumberColumn("Unidades", format="%.0f"),
                "TRANSACCIONES": st.column_config.NumberColumn("Transacciones", format="%d"),
                "TICKET": st.column_config.NumberColumn("Ticket prom.", format="%.2f"),
            },
        )

# ----------------------------- Productos ------------------------------
with tab_prod:
    medida = st.radio("Medir por", ["Ventas (S/)", "Unidades"], horizontal=True)
    col_m = ID_TOTAL if medida.startswith("Ventas") else ID_CANTIDAD

    prod = f.groupby([ID_TIPO_ARTICULO, ID_ARTICULO])[col_m].sum().reset_index()
    prod = prod[prod[col_m] > 0]
    fig = px.treemap(prod, path=[ID_TIPO_ARTICULO, ID_ARTICULO], values=col_m,
                     title=f"Familias y productos ({medida})")
    mostrar(fig, 460)

    izq, der = st.columns(2)
    with izq:
        top = prod.nlargest(10, col_m).sort_values(col_m)
        fig = px.bar(top, x=col_m, y=ID_ARTICULO, orientation="h", color=ID_TIPO_ARTICULO,
                     title="Top 10 productos", labels={col_m: medida, ID_ARTICULO: ""})
        mostrar(fig, 420)
    with der:
        nivel = st.radio("Agrupar por", ["Zona", "Distrito"], horizontal=True)
        eje = ID_ZONA if nivel == "Zona" else ID_DISTRITO
        comb = f.groupby([eje, ID_TIPO_ARTICULO])[col_m].sum().reset_index()
        fig = px.bar(comb, x=eje, y=col_m, color=ID_TIPO_ARTICULO, barmode="stack",
                     title=f"{medida} por {nivel.lower()} y familia", labels={col_m: medida})
        mostrar(fig, 420)

# ----------------------------- Datos ----------------------------------
with tab_datos:
    columnas = [ID_FECHA, ID_HORA, ID_LOCAL, ID_DISTRITO, ID_ZONA, ID_ARTICULO,
                ID_TIPO_ARTICULO, ID_CANTIDAD, ID_TOTAL]
    st.caption(f"{len(f):,} filas con los filtros actuales (se muestran las primeras 500).")
    st.dataframe(f[columnas].head(500), hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        f[columnas].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ventas_{mes}.csv",
        mime="text/csv",
    )
