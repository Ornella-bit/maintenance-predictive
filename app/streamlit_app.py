from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DOSSIER_APP = Path(__file__).resolve().parent
DOSSIER_PROJET = DOSSIER_APP.parent

st.set_page_config(page_title="Maintenance prédictive", layout="wide")


@st.cache_resource
def charger_modele():
    return joblib.load(DOSSIER_PROJET / "models" / "rul_model.joblib")


@st.cache_data
def charger_donnees():
    return pd.read_csv(DOSSIER_APP / "donnees_demo.csv")


def ajouter_features(df, capteurs, fenetre=5):
    df = df.sort_values(["moteur", "cycle"]).copy()
    groupes = df.groupby("moteur")[capteurs]
    moy = groupes.rolling(fenetre, min_periods=1).mean().reset_index(level=0, drop=True)
    ecart = groupes.rolling(fenetre, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    moy.columns = [f"{c}_moy" for c in capteurs]
    ecart.columns = [f"{c}_std" for c in capteurs]
    return pd.concat([df, moy, ecart], axis=1)


paquet = charger_modele()
modele = paquet["modele"]
features = paquet["features"]
capteurs = paquet["capteurs"]
donnees = charger_donnees()

st.title("Maintenance prédictive : durée de vie restante d'un moteur")
st.write(
    "Démonstration sur le jeu de données NASA C-MAPSS (FD001). "
    "Le modèle estime le nombre de cycles avant la panne (RUL)."
)

# Barre latérale : choix du moteur et du cycle
st.sidebar.header("Paramètres")
moteur = st.sidebar.selectbox("Moteur", sorted(donnees["moteur"].unique().tolist()))
df_moteur = donnees[donnees["moteur"] == moteur]
cycle_max = int(df_moteur["cycle"].max())
cycle = st.sidebar.slider("Cycle observé", min_value=1, max_value=cycle_max, value=cycle_max)

# Prédiction au cycle choisi
df_f = ajouter_features(df_moteur, capteurs)
ligne = df_f[df_f["cycle"] == cycle]
rul_predite = max(float(modele.predict(ligne[features])[0]), 0.0)
rul_reelle = min(float(ligne["RUL_reelle"].iloc[0]), 125.0)

col1, col2, col3 = st.columns(3)
col1.metric("RUL prédite (cycles)", f"{rul_predite:.0f}")
col2.metric("RUL réelle (plafonnée à 125)", f"{rul_reelle:.0f}")
col3.metric("Cycle observé", f"{cycle} / {cycle_max}")

# Indicateur de statut
if rul_predite < 30:
    st.error("Statut ROUGE : panne proche, maintenance à planifier immédiatement.")
elif rul_predite < 60:
    st.warning("Statut ORANGE : usure avancée, maintenance à planifier.")
else:
    st.success("Statut VERT : fonctionnement normal.")

# Courbes des capteurs clés
st.subheader("Capteurs clés jusqu'au cycle observé")
historique = df_moteur[df_moteur["cycle"] <= cycle]
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, capteur in zip(axes, ["s4", "s9", "s11", "s7"]):
    ax.plot(historique["cycle"], historique[capteur])
    ax.set_title(capteur)
    ax.set_xlabel("cycle")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# RUL prédite et réelle au fil des cycles
st.subheader("RUL prédite et réelle au fil des cycles")
hist_f = df_f[df_f["cycle"] <= cycle]
pred_hist = modele.predict(hist_f[features]).clip(min=0)
fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.plot(hist_f["cycle"], hist_f["RUL_reelle"].clip(upper=125), label="RUL réelle (plafonnée à 125)")
ax2.plot(hist_f["cycle"], pred_hist, label="RUL prédite")
ax2.set_xlabel("cycle")
ax2.set_ylabel("RUL")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.caption("Projet pédagogique : ne pas utiliser pour des décisions de maintenance réelles.")
