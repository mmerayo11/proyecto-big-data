import os
import streamlit as st
import pandas as pd
import altair as alt

# 1. Configuración inicial de la página
st.set_page_config(
    page_title="Dashboard Demanda Eléctrica",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data
def cargar_datos_clima():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_parquet = os.path.join(ruta_script, "..", "data_lake", "plata", "aemet_limpio.parquet")
    
    try:
        df = pd.read_parquet(ruta_parquet)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        columnas_temperatura = ['tmax', 'tmin', 'tmed']
        for col in columnas_temperatura:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except FileNotFoundError:
        st.error(f"Archivo no encontrado. Ruta buscada: {ruta_parquet}")
        return pd.DataFrame()

df_clima = cargar_datos_clima()

# 2. DISEÑO DE LA BARRA LATERAL (Filtros Avanzados)
st.sidebar.header("⚙️ Controles del Dashboard")

if not df_clima.empty:
    # --- Selector Múltiple de Territorios ---
    provincias_disponibles = sorted(df_clima['provincia'].unique().tolist())
    opciones_territorios = ["🇪🇸 Media Nacional"] + provincias_disponibles
    
    # Valores por defecto lógicos para la primera carga
    seleccion_territorios_defecto = ["🇪🇸 Media Nacional"]
    for variante_leon in ["LEON", "León", "León/Leon"]:
        if variante_leon in provincias_disponibles:
            seleccion_territorios_defecto.append(variante_leon)
            break

    territorios_seleccionados = st.sidebar.multiselect(
        "📍 Selecciona Territorios:",
        options=opciones_territorios,
        default=seleccion_territorios_defecto
    )
    
    # --- NUEVO: Selector Múltiple de Métricas de Temperatura ---
    metricas_opciones = {
        "Máxima (tmax)": "tmax",
        "Media (tmed)": "tmed",
        "Mínima (tmin)": "tmin"
    }
    
    metricas_seleccionadas = st.sidebar.multiselect(
        "🌡️ Muestreo de Temperaturas:",
        options=list(metricas_opciones.keys()),
        default=["Media (tmed)"] # Por defecto arranca mostrando solo las medias
    )
    
    # Mapeamos los nombres legibles a los nombres reales de las columnas del DataFrame
    columnas_metricas_reales = [metricas_opciones[m] for m in metricas_seleccionadas]
    
    st.sidebar.divider()
    
    # --- Filtros Temporales ---
    años_disponibles = sorted(df_clima['fecha'].dt.year.unique())
    año_seleccionado = st.sidebar.selectbox(
        "📅 Selecciona Año:", 
        ["Todos los años"] + list(años_disponibles),
        index=0
    )
    
    if año_seleccionado != "Todos los años":
        meses_nombres = ["Todos los meses", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        mes_seleccionado = st.sidebar.selectbox("📆 Selecciona Mes:", meses_nombres, index=0)
    else:
        mes_seleccionado = "Todos los meses"
        
    st.sidebar.divider()
    
    # --- Granularidad Temporal ---
    opciones_agrupacion = ["Diario", "Semanal", "Mensual", "Anual"]
    if mes_seleccionado != "Todos los meses":
        opciones_agrupacion = ["Diario", "Semanal"]
        
    agrupacion = st.sidebar.selectbox("⏱️ Agrupar datos por:", opciones_agrupacion, index=0)
else:
    territorios_seleccionados = []
    columnas_metricas_reales = []

# 3. LÓGICA DE PROCESAMIENTO DINÁMICA
df_final_plot = pd.DataFrame()
df_tabla = pd.DataFrame()

if not df_clima.empty and territorios_seleccionados and columnas_metricas_reales:
    # Filtro de tiempo general
    df_tiempo = df_clima.copy()
    if año_seleccionado != "Todos los años":
        df_tiempo = df_tiempo[df_tiempo['fecha'].dt.year == año_seleccionado]
        if mes_seleccionado != "Todos los meses":
            num_mes = meses_nombres.index(mes_seleccionado)
            df_tiempo = df_tiempo[df_tiempo['fecha'].dt.month == num_mes]

    lista_dfs_plot = []
    provincias_reales = [t for t in territorios_seleccionados if t != "🇪🇸 Media Nacional"]
    
    # Conservamos registros brutos en la tabla para las provincias seleccionadas
    if provincias_reales:
        df_tabla = df_tiempo[df_tiempo['provincia'].isin(provincias_reales)]

    # Procesamos de forma independiente cada territorio mapeando sus métricas
    for territorio in territorios_seleccionados:
        if territorio == "🇪🇸 Media Nacional":
            df_temp = df_tiempo.groupby('fecha')[['tmax', 'tmin', 'tmed']].mean()
            nombre_linea = "Media Nacional"
        else:
            df_temp = df_tiempo[df_tiempo['provincia'] == territorio].groupby('fecha')[['tmax', 'tmin', 'tmed']].mean()
            nombre_linea = territorio

        # Aplicar agregación por tiempo
        if agrupacion == "Semanal": df_temp = df_temp.resample('W').mean()
        elif agrupacion == "Mensual": df_temp = df_temp.resample('ME').mean()
        elif agrupacion == "Anual": df_temp = df_temp.resample('YE').mean()
            
        df_temp = df_temp.reset_index()

        # Transformación estructural (Melt) basada únicamente en las métricas activas
        df_melt = df_temp.melt(id_vars=['fecha'], value_vars=columnas_metricas_reales, var_name='Métrica', value_name='Grados')
        
        # Formateamos el nombre de la serie para la leyenda (Ej: "tmax (Media Nacional)")
        df_melt['Serie'] = df_melt['Métrica'] + f" ({nombre_linea})"
        lista_dfs_plot.append(df_melt)

    if lista_dfs_plot:
        df_final_plot = pd.concat(lista_dfs_plot, ignore_index=True)


# 4. DISEÑO DEL CUERPO PRINCIPAL
st.title("⚡ Predictor de Demanda para Redes Eléctricas")
st.markdown("Proyecto Integrador - Curso de Especialización IA y Big Data (IES San Andrés)")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Tabla de Registros")
    if not df_tabla.empty:
        st.dataframe(df_tabla.head(15), use_container_width=True)
    elif "🇪🇸 Media Nacional" in territorios_seleccionados and len(territorios_seleccionados) == 1:
        st.info("Visualizando agregación nacional pura. Añada una provincia para poblar la tabla informativa.")
    else:
        st.warning("Seleccione territorio o active filtros para renderizar registros.")

with col2:
    st.subheader(f"📈 Matriz de Comparación Climatológica ({agrupacion})")
    
    if not df_final_plot.empty:
        meses_es = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        
        # Formateo semántico del eje temporal
        if agrupacion == "Mensual":
            df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: f"{meses_es[x.month]} '{x.strftime('%y')}")
        elif agrupacion in ["Diario", "Semanal"]:
            df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: f"{x.day} {meses_es[x.month]} '{x.strftime('%y')}")
        elif agrupacion == "Anual":
            df_final_plot['Etiqueta'] = df_final_plot['fecha'].apply(lambda x: str(x.year))
            
        df_final_plot = df_final_plot.sort_values('fecha')
        orden_correcto = df_final_plot['Etiqueta'].unique().tolist()
        
        # Construcción interactiva con Altair
        grafico = alt.Chart(df_final_plot).mark_line().encode(
            x=alt.X('Etiqueta:N', sort=orden_correcto, title=None),
            y=alt.Y('Grados:Q', title='Temperatura (°C)', scale=alt.Scale(zero=False)),
            color=alt.Color('Serie:N', title='Series Activas'),
            tooltip=['Etiqueta', 'Serie', 'Grados']
        ).interactive()
        
        st.altair_chart(grafico, use_container_width=True)
    else:
        st.warning("Seleccione al menos un Territorio y una Métrica en la barra de controles.")

st.divider()
st.subheader("💡 Relación con la Demanda Eléctrica (Capa Oro)")
st.info("Arquitectura preparada para inyectar y cruzar las series seleccionadas con las curvas de consumo energético.")