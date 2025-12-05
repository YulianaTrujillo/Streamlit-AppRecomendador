import pandas as pd
import numpy as np
import joblib
import requests
import os
import streamlit as st
# ------------------------------------------------
# CARGA DEL MODELO Y LOS DATOS
# --------------------------------------
# Rutas locales donde app.py va a guardar los archivos
DF_PATH = "data/df_weighted.csv"
MODEL_PATH = "model/nn_cosine_model.joblib"

def load_recommender():
  
    df = pd.read_csv(DF_PATH)
    nn = joblib.load(MODEL_PATH)

    # Selección de variables
    feature_cols = [
        c for c in df.columns if c not in
        ["track_id", "track_name", "artists", "album_name", "track_genre"]
    ]

    X_emb = df[feature_cols].to_numpy()

      return df, nn, X_emb
# ------------------------------------------------
# FUNCIÓN DE RECOMENDACIÓN PRINCIPAL
# ------------------------------------------------
#Función principal para elegir el top de canciones dado de track_id de una canción
def recommend_by_track_id(track_id, nn, df, X_emb, k_fixed=3, k_random=7, top_N=50, seed=None):

    """
      - k_fixed: top más similares, ordenados por similitud (deterministas)
      - k_random: canciones tomadas aleatoriamente del top_N
      - top_N: número de candidatos del cual muestrearemos los aleatorios

    total top_k = k_fixed + k_random (por defecto = 10)
    """

    # --- 1. Buscar índice por track_id ---
    mask = df["track_id"] == track_id
    if not mask.any():
        raise ValueError(f"track_id '{track_id}' no existe en el dataset.")

    index = df.index[mask][0]        # label del df
    pos = df.index.get_loc(index)    # posición en X_emb

    # --- 2. Vector de características ---
    vec = X_emb[pos].reshape(1, -1)    
    n_retrieve = max(top_N, k_fixed + k_random)


    # --- 3. Obtener vecinos ---
    distances, indices = nn.kneighbors(vec, n_neighbors=n_retrieve)
    distances = distances.flatten()
    indices = indices.flatten()

    # --- convertir distancia a similitud
    metric = getattr(nn, "effective_metric_", getattr(nn, "metric", None))
    if metric == "cosine":
        similitudes = 1 - distances
    else:
        similitudes = 1 / (1 + distances)

    # --- 4. Crear recomendaciones ---
    resultados = []
    for dist, p, sim in zip(distances, indices, similitudes):
        row_label = df.index[p]
        tid = df.loc[row_label, "track_id"]
        if tid == track_id: 
           continue
        resultados.append({
            "track_id": tid,
            "track_name": df.loc[row_label, "track_name"],
            "artists": df.loc[row_label, "artists"],
            "track_genre": df.loc[row_label, "track_genre"],
            "similitud": round(float(sim), 4)
        })

    # --- Top final limpio ---
    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df.drop_duplicates(subset=["track_name", "artists"], keep="first").reset_index(drop=True)
    #resultados_df = resultados_df.head(top_k).reset_index(drop=True)

    # asegurar que hay suficientes candidatos
    if resultados_df.empty:
        return resultados_df

    # ORDENAR por similitud
    resultados_df = resultados_df.sort_values(by="similitud", ascending=False).reset_index(drop=True)

    # --- 1) TOP fijos ---
    fixed_part = resultados_df.head(k_fixed)

    # --- 2) RANDOM desde el resto del top_N ---
    rest_pool = resultados_df.iloc[k_fixed:top_N]

    rng = np.random.default_rng(seed)

    if len(rest_pool) >= k_random:
        random_idx = rng.choice(rest_pool.index, size=k_random, replace=False)
        random_part = resultados_df.loc[random_idx]
    else:
        # fallback: tomar todo lo que haya
        random_part = rest_pool

    # unir ambas partes
    final_df = pd.concat([fixed_part, random_part]).reset_index(drop=True)


    return final_df



