import streamlit as st
import pandas as pd
import os
import requests

MODEL_URL = st.secrets["MODEL_URL"]
DF_URL = st.secrets["DF_URL"]

MODEL_PATH = "model/nn_cosine_model.joblib"
DF_PATH = "data/df_weighted.csv"


@st.cache_resource
def ensure_files():
    # Descargar modelo si no existe
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Descargar DF si no existe
    if not os.path.exists(DF_PATH):
        os.makedirs(os.path.dirname(DF_PATH), exist_ok=True)
        with requests.get(DF_URL, stream=True) as r:
            r.raise_for_status()
            with open(DF_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    return MODEL_PATH, DF_PATH


@st.cache_resource
def load_recommender():
    # Asegurarnos de que los archivos están descargados
    ensure_files()

    # Importar recommender DESPUÉS de tener los archivos
    from recommender import df, nn, X_emb, recommend_by_track_id

    return df, nn, X_emb, recommend_by_track_id



st.set_page_config(page_title="Music Recommender", layout="centered")

st.title("Recomendador de Música")
st.write("Busca una canción y obtén recomendaciones similares usando tu modelo optimizado.")


# ------------------------------------------------
# BUSCADOR DE CANCIONES
# ------------------------------------------------
st.subheader("Buscar canción")

query = st.text_input("Escribe parte del nombre de la canción:")

selected_track_id = None

if query:
    df_ui = df[['track_id','track_name','artists','track_genre']].drop_duplicates(subset=['track_name','artists'])
    filtradas = df_ui[df_ui["track_name"].str.contains(query, case=False, na=False)]

    #filtradas = df[df["track_name"].str.contains(query, case=False, na=False)]

    if filtradas.empty:
        st.warning("No se encontraron canciones.")
    else:
        st.success(f"Se encontraron {len(filtradas)} canciones")

        # Crear una columna combinando nombre y artista
        filtradas["display_name"] = filtradas["track_name"] + " — " + filtradas["artists"] + " - " + filtradas["track_genre"]  

        # Crear selectbox para que el usuario elija una
        selected_song = st.selectbox(
            "Selecciona la canción:",
            options=filtradas["display_name"].tolist()
        )

        # Obtener track_id interno según la selección
        selected_track_id = filtradas.loc[
            filtradas["display_name"] == selected_song, "track_id"
        ].values[0]


        st.info(f"Track ID seleccionado automáticamente: `{selected_track_id}`")
         # Mostrar info de la canción base
        row = df[df["track_id"] == selected_track_id].iloc[0]
        st.subheader("🎧 Canción seleccionada")
        st.write(f"**{row['track_name']}** — {row['artists']}")
        st.write(f"**Género:** {row['track_genre']}")
        st.write("---")
        
# ------------------------------------------------
# BOTÓN PARA RECOMENDAR
# ------------------------------------------------
st.subheader("Obtener recomendaciones")

if st.button("Recomendar"):
    if not selected_track_id:
        st.error("Debes buscar una canción y seleccionarla.")
    else:
        try:
            recs = recommend_by_track_id(selected_track_id, nn, df, X_emb,k_fixed=3, k_random=7, top_N=50)

            st.success("Recomendaciones generadas:")

            st.dataframe(recs, height=350)

        except Exception as e:
            st.error(f"Error: {str(e)}")

