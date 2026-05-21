import os
import streamlit as st
import pandas as pd
import altair as alt

# Configuración inicial de la página
st.set_page_config(
    page_title="Dashboard Demanda Eléctrica",
    page_icon="⚡",
    layout="wide"
)

# CARGA DE DATOS
@st.cache_data
def cargar_datos_clima():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_parquet = os.path.join(ruta_script, "..", "data_lake", "plata", "aemet_limpio.parquet")
    try:
        df = pd.read_parquet(ruta_parquet)
        df['fecha'] = pd.to_datetime(df['fecha'])
        for col in ['tmax', 'tmin', 'tmed']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def cargar_capa_oro():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_parquet = os.path.join(ruta_script, "..", "data_lake", "oro", "dataset_analitico.parquet")
    try:
        df = pd.read_parquet(ruta_parquet)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_clima = cargar_datos_clima()
df_oro = cargar_capa_oro()

# DISEÑO DE LA BARRA LATERAL
st.sidebar.header("⚙️ Controles del Dashboard")

if not df_clima.empty:
    provincias_disponibles = sorted(df_clima['provincia'].unique().tolist())
    opciones_territorios = ["🇪🇸 Media Nacional"] + provincias_disponibles
    
    seleccion_territorios_defecto = ["🇪🇸 Media Nacional"]
    for variante in ["LEON", "León", "León/Leon"]:
        if variante in provincias_disponibles:
            seleccion_territorios_defecto.append(variante)
            break

    territorios_seleccionados = st.sidebar.multiselect("📍 Selecciona Territorios:", opciones_territorios, default=seleccion_territorios_defecto)
    
    metricas_opciones = {"Máxima (tmax)": "tmax", "Media (tmed)": "tmed", "Mínima (tmin)": "tmin"}
    metricas_seleccionadas = st.sidebar.multiselect("🌡️ Muestreo de Temperaturas:", list(metricas_opciones.keys()), default=["Media (tmed)"])
    columnas_metricas_reales = [metricas_opciones[m] for m in metricas_seleccionadas]
    
    st.sidebar.divider()
    
    años_disponibles = sorted(df_clima['fecha'].dt.year.unique())
    año_seleccionado = st.sidebar.selectbox("📅 Selecciona Año:", ["Todos los años"] + list(años_disponibles), index=0)
    
    if año_seleccionado != "Todos los años":
        meses_nombres = ["Todos los meses", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        mes_seleccionado = st.sidebar.selectbox("📆 Selecciona Mes:", meses_nombres, index=0)
    else:
        mes_seleccionado = "Todos los meses"
        
    st.sidebar.divider()
    
    opciones_agrupacion = ["Diario", "Semanal", "Mensual", "Anual"]
    if mes_seleccionado != "Todos los meses": opciones_agrupacion = ["Diario", "Semanal"]
    agrupacion = st.sidebar.selectbox("⏱️ Agrupar datos por:", opciones_agrupacion, index=0)
else:
    territorios_seleccionados, columnas_metricas_reales = [], []

# LÓGICA DE PROCESAMIENTO (CLIMA Y ORO)
df_final_plot = pd.DataFrame()
df_tabla = pd.DataFrame()
df_oro_filtrado = pd.DataFrame()

if not df_clima.empty and territorios_seleccionados and columnas_metricas_reales:
    # Filtro temporal para Clima
    df_tiempo = df_clima.copy()
    if año_seleccionado != "Todos los años":
        df_tiempo = df_tiempo[df_tiempo['fecha'].dt.year == año_seleccionado]
        if mes_seleccionado != "Todos los meses":
            num_mes = meses_nombres.index(mes_seleccionado)
            df_tiempo = df_tiempo[df_tiempo['fecha'].dt.month == num_mes]

    # Filtro temporal idéntico para Capa Oro
    if not df_oro.empty:
        df_oro_filtrado = df_oro.copy()
        if año_seleccionado != "Todos los años":
            df_oro_filtrado = df_oro_filtrado[df_oro_filtrado['fecha'].dt.year == año_seleccionado]
            if mes_seleccionado != "Todos los meses":
                df_oro_filtrado = df_oro_filtrado[df_oro_filtrado['fecha'].dt.month == num_mes]

    lista_dfs_plot = []
    provincias_reales = [t for t in territorios_seleccionados if t != "🇪🇸 Media Nacional"]
    if provincias_reales: df_tabla = df_tiempo[df_tiempo['provincia'].isin(provincias_reales)]

    for territorio in territorios_seleccionados:
        if territorio == "🇪🇸 Media Nacional":
            df_temp = df_tiempo.groupby('fecha')[['tmax', 'tmin', 'tmed']].mean()
            nombre_linea = "Media Nacional"
        else:
            df_temp = df_tiempo[df_tiempo['provincia'] == territorio].groupby('fecha')[['tmax', 'tmin', 'tmed']].mean()
            nombre_linea = territorio

        if agrupacion == "Semanal": df_temp = df_temp.resample('W').mean()
        elif agrupacion == "Mensual": df_temp = df_temp.resample('ME').mean()
        elif agrupacion == "Anual": df_temp = df_temp.resample('YE').mean()
            
        df_temp = df_temp.reset_index()
        df_melt = df_temp.melt(id_vars=['fecha'], value_vars=columnas_metricas_reales, var_name='Métrica', value_name='Grados')
        df_melt['Serie'] = df_melt['Métrica'] + f" ({nombre_linea})"
        lista_dfs_plot.append(df_melt)

    if lista_dfs_plot: df_final_plot = pd.concat(lista_dfs_plot, ignore_index=True)


# DISEÑO DEL CUERPO PRINCIPAL
st.title("⚡ Predictor de Demanda para Redes Eléctricas")
st.markdown("Proyecto Integrador - Curso de Especialización IA y Big Data (IES San Andrés)")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Tabla de Registros (Clima)")
    if not df_tabla.empty: st.dataframe(df_tabla.head(15), use_container_width=True)
    elif "🇪🇸 Media Nacional" in territorios_seleccionados and len(territorios_seleccionados) == 1:
        st.info("Visualizando agregación nacional pura. Añada una provincia para poblar la tabla.")
    else: st.warning("Seleccione territorio o active filtros para renderizar registros.")

with col2:
    st.subheader(f"📈 Matriz de Comparación Climatológica ({agrupacion})")
    if not df_final_plot.empty:
        meses_es = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        if agrupacion == "Mensual": df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: f"{meses_es[x.month]} '{x.strftime('%y')}")
        elif agrupacion in ["Diario", "Semanal"]: df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: f"{x.day} {meses_es[x.month]} '{x.strftime('%y')}")
        elif agrupacion == "Anual": df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: str(x.year))
            
        df_final_plot = df_final_plot.sort_values('fecha')
        orden_correcto = df_final_plot['Etiqueta'].unique().tolist()
        
        grafico_clima = alt.Chart(df_final_plot).mark_line().encode(
            x=alt.X('Etiqueta:N', sort=orden_correcto, title=None),
            y=alt.Y('Grados:Q', title='Temperatura (°C)', scale=alt.Scale(zero=False)),
            color=alt.Color('Serie:N', title='Series Activas'),
            tooltip=['Etiqueta', 'Serie', 'Grados']
        ).interactive()
        st.altair_chart(grafico_clima, use_container_width=True)
    else: st.warning("Faltan datos climatológicos.")

st.divider()
st.subheader("💡 Capa Oro: Demanda Eléctrica e Impacto de Festivos")

if not df_oro_filtrado.empty:
    
    # Tarjetas KPI de Festivos
    consumo_laboral = df_oro_filtrado[df_oro_filtrado['es_festivo'] == 0]['demanda_mwh'].mean()
    consumo_festivo = df_oro_filtrado[df_oro_filtrado['es_festivo'] == 1]['demanda_mwh'].mean()
    
    st.markdown("##### Resumen de Consumo Medio (Según rango seleccionado)")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("⚡ Día Laborable", f"{consumo_laboral:,.0f} MWh" if pd.notnull(consumo_laboral) else "N/A")
    kpi2.metric("🎉 Día Festivo", f"{consumo_festivo:,.0f} MWh" if pd.notnull(consumo_festivo) else "N/A")
    
    if pd.notnull(consumo_laboral) and pd.notnull(consumo_festivo) and consumo_laboral > 0:
        caida = ((consumo_laboral - consumo_festivo) / consumo_laboral) * 100
        kpi3.metric("📉 Desplome Energético en Festivos", f"-{caida:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos de Demanda
    col_demanda1, col_demanda2 = st.columns(2)
    
    with col_demanda1:
        st.markdown(f"**Evolución de la Demanda Nacional ({agrupacion})**")
        df_oro_grp = df_oro_filtrado.set_index('fecha')
        
        # Aplicamos la misma agrupación que eligió el usuario
        if agrupacion == "Semanal": df_oro_grp = df_oro_grp.resample('W').mean()
        elif agrupacion == "Mensual": df_oro_grp = df_oro_grp.resample('ME').mean()
        elif agrupacion == "Anual": df_oro_grp = df_oro_grp.resample('YE').mean()
        
        st.line_chart(df_oro_grp['demanda_mwh'])
        
    with col_demanda2:
        st.markdown("**Correlación: Temperatura vs Demanda**")
        
        # Preparamos los datos para el gráfico de dispersión
        df_scatter = df_oro_filtrado.copy()
        df_scatter['Tipo de Día'] = df_scatter['es_festivo'].map({0: 'Laborable', 1: 'Festivo'})
        
        # Gráfico de dispersión Altair
        scatter = alt.Chart(df_scatter).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('tmed:Q', title='Temp. Media 🇪🇸 (°C)', scale=alt.Scale(zero=False)),
            y=alt.Y('demanda_mwh:Q', title='Demanda (MWh)', scale=alt.Scale(zero=False)),
            color=alt.Color('Tipo de Día:N', scale=alt.Scale(domain=['Laborable', 'Festivo'], range=['#1f77b4', '#d62728'])),
            tooltip=[alt.Tooltip('fecha:T', title='Fecha'), 'tmed:Q', 'demanda_mwh:Q', 'Tipo de Día:N']
        ).interactive()
        
        st.altair_chart(scatter, use_container_width=True)
        st.caption("Nota: Se muestra el detalle diario para aislar claramente los puntos rojos (festivos).")

else:
    st.warning("No hay datos en la Capa Oro. Asegúrate de ejecutar pipeline_oro.py primero.")