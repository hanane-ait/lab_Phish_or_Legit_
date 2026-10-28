
from pathlib import Path
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Phish or Legit?",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_assets():

    required_files = [
        ART / "models.joblib",
        ART / "metrics.json",
        ART / "test_examples.csv",
    ]

    # Entraîner les modèles si les résultats n'existent pas.
    if not all(path.exists() for path in required_files):

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_model.py")
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            details = result.stderr[-3000:] or result.stdout[-3000:]

            raise RuntimeError(
                details or "L'entraînement du modèle a échoué."
            )

    models = joblib.load(ART / "models.joblib")

    saved_metrics = json.loads(
        (ART / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    examples = pd.read_csv(
        ART / "test_examples.csv"
    )

    return models, saved_metrics, examples


def contributions(pipeline, text, model_key, limit=12):
    """
    Calcule les contributions approximatives des caractéristiques TF-IDF.

    Une contribution positive favorise la classe phishing.
    Une contribution négative favorise la classe légitime.
    """

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    vector = vectorizer.transform([text])
    values = vector.toarray()[0]

    active_indices = np.flatnonzero(values)

    if len(active_indices) == 0:
        return pd.DataFrame(
            columns=[
                "Caractéristique",
                "Contribution",
                "Interprétation",
            ]
        )

    feature_names = vectorizer.get_feature_names_out()

    if model_key == "logistic":
        weights = classifier.coef_[0]
    else:
        weights = (
            classifier.feature_log_prob_[1]
            - classifier.feature_log_prob_[0]
        )

    contribution_values = (
        values[active_indices] * weights[active_indices]
    )

    order = np.argsort(
        np.abs(contribution_values)
    )[::-1][:limit]

    rows = []

    for position in order:

        contribution = float(
            contribution_values[position]
        )

        rows.append({
            "Caractéristique": feature_names[
                active_indices[position]
            ],
            "Contribution": contribution,
            "Interprétation": (
                "Augmente le score phishing"
                if contribution > 0
                else "Diminue le score phishing"
            ),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 1. Titre et avertissement
# --------------------------------------------------

st.title("🛡️ Phish or Legit?")

st.caption(
    "Classification de messages en anglais avec TF-IDF, "
    "régression logistique et Naive Bayes."
)

st.warning(
    "Aide à la décision uniquement : l'application ne bloque ni ne "
    "supprime de message. Le corpus est public, équilibré et en anglais. "
    "Les scores ne garantissent pas la détection de toutes les attaques."
)


# --------------------------------------------------
# 2. Charger les modèles
# --------------------------------------------------

try:
    models, saved, examples = load_assets()

except Exception as exc:
    st.error(f"Préparation impossible : {exc}")

    st.info(
        "Vérifie ta connexion Internet et lance "
        "`python train_model.py` dans le terminal."
    )

    st.stop()


# --------------------------------------------------
# 3. Choisir le modèle et son seuil
# --------------------------------------------------

model_label = st.selectbox(
    "Modèle",
    [
        "Régression logistique",
        "Naive Bayes multinomial",
    ],
)

model_key = (
    "logistic"
    if model_label == "Régression logistique"
    else "naive_bayes"
)

model = models[model_key]

# Le seuil choisi sur validation devient le seuil initial.
default_threshold = float(
    saved[model_key]["threshold_selected_on_validation"]
)

threshold = st.slider(
    "Seuil de décision — la prédiction change en direct",
    min_value=0.05,
    max_value=0.95,
    value=default_threshold,
    step=0.01,
    key=f"threshold_{model_key}",
)

st.caption(
    f"Seuil sélectionné sur validation : {default_threshold:.2f}. "
    "Tu peux déplacer le curseur pour observer l'effet sur la décision."
)


# --------------------------------------------------
# 4. Choisir un exemple ou saisir un message
# --------------------------------------------------

example_names = (
    examples["example_name"]
    .fillna("Exemple du test")
    .tolist()
)

options = ["Écrire mon propre message"] + example_names

choice = st.selectbox(
    "Exemples issus du jeu de test",
    options,
)

default_text = ""

if choice != options[0]:

    matching_rows = examples.loc[
        examples["example_name"] == choice,
        "text"
    ]

    if not matching_rows.empty:
        default_text = str(matching_rows.iloc[0])


# Initialiser l'état de la zone de texte.
if "message_input" not in st.session_state:
    st.session_state["message_input"] = default_text

# Charger le texte correspondant à l'exemple sélectionné.
if choice != options[0]:
    st.session_state["message_input"] = default_text

message = st.text_area(
    "Message à analyser",
    key="message_input",
    height=170,
    placeholder=(
        "Colle ici le contenu d'un email ou d'un message..."
    ),
)

analyse = st.button(
    "Classifier le message",
    type="primary",
)

if analyse:
    st.session_state["analysed_message"] = message

text_to_analyse = st.session_state.get(
    "analysed_message",
    ""
)

if analyse and not message.strip():
    st.warning(
        "Saisis ou sélectionne un message avant de lancer l'analyse."
    )


# --------------------------------------------------
# 5. Calcul du score et de la prédiction
# --------------------------------------------------

if text_to_analyse and text_to_analyse.strip():

    score = float(
        model.predict_proba([text_to_analyse])[0, 1]
    )

    prediction = int(score >= threshold)

    col_a, col_b, col_c = st.columns(3)

    col_a.metric(
        "Classe prédite",
        "PHISHING" if prediction else "LÉGITIME",
    )

    col_b.metric(
        "Score phishing du modèle",
        f"{score:.1%}",
    )

    col_c.metric(
        "Seuil actif",
        f"{threshold:.0%}",
    )

    st.progress(
        min(max(score, 0.0), 1.0)
    )

    st.caption(
        "Le score est estimé par le modèle. Il ne représente pas "
        "nécessairement une probabilité parfaitement calibrée, "
        "notamment pour Naive Bayes."
    )

    if prediction:
        st.error(
            "Message potentiellement suspect. Vérifie l'expéditeur "
            "et les liens par un canal indépendant."
        )
    else:
        st.success(
            "Le modèle classe ce message comme légitime. "
            "Cela ne prouve pas qu'il est sûr."
        )

    # --------------------------------------------------
    # 6. Explication du score
    # --------------------------------------------------

    st.subheader("Explication par caractéristiques")

    explanation = contributions(
        model,
        text_to_analyse,
        model_key,
    )

    if explanation.empty:

        st.info(
            "Aucune caractéristique connue du vocabulaire TF-IDF "
            "n'a été trouvée."
        )

    else:

        st.dataframe(
            explanation.style.format({
                "Contribution": "{:+.4f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Pour la régression logistique, la contribution est "
            "approximativement la valeur TF-IDF multipliée par le "
            "coefficient appris. Pour Naive Bayes, elle repose sur "
            "la différence des log-probabilités des classes."
        )


# --------------------------------------------------
# 7. Évaluation finale sur tout le jeu de test
# --------------------------------------------------

st.divider()

st.subheader(
    "Évaluation finale sur l'ensemble du jeu de test"
)

final_metrics = saved[model_key][
    "test_metrics_at_selected_threshold"
]

final_cm = saved[model_key][
    "test_confusion_matrix_at_selected_threshold"
]

st.caption(
    "Ces résultats sont calculés sur l'ensemble du jeu de test "
    "indépendant, avec le seuil sélectionné sur validation. "
    "Ils restent fixes lorsque le curseur change."
)

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Précision",
    f"{final_metrics['precision']:.3f}",
)

metric_columns[1].metric(
    "Rappel",
    f"{final_metrics['recall']:.3f}",
)

metric_columns[2].metric(
    "F1-score",
    f"{final_metrics['f1']:.3f}",
)

metric_columns[3].metric(
    "Brier score",
    f"{final_metrics['brier']:.3f}",
)

st.write(
    "Matrice de confusion au seuil sélectionné sur validation :"
)

st.dataframe(
    pd.DataFrame(
        final_cm,
        index=[
            "Réel légitime",
            "Réel phishing",
        ],
        columns=[
            "Prédit légitime",
            "Prédit phishing",
        ],
    ),
    use_container_width=True,
)


# --------------------------------------------------
# 8. Comparaison des deux modèles
# --------------------------------------------------

st.subheader("Comparaison des modèles")

comparison_rows = []

for key, label in [
    ("logistic", "Régression logistique"),
    ("naive_bayes", "Naive Bayes multinomial"),
]:

    model_result = saved[key]

    standard = model_result["test_metrics_at_0_5"]

    selected = model_result[
        "test_metrics_at_selected_threshold"
    ]

    comparison_rows.append({
        "Modèle": label,
        "Seuil sélectionné": (
            model_result["threshold_selected_on_validation"]
        ),
        "F1 test à 0.50": standard["f1"],
        "F1 test au seuil sélectionné": selected["f1"],
        "Précision au seuil sélectionné": selected["precision"],
        "Rappel au seuil sélectionné": selected["recall"],
        "Brier score": selected["brier"],
    })

comparison_df = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison_df.style.format({
        "Seuil sélectionné": "{:.2f}",
        "F1 test à 0.50": "{:.3f}",
        "F1 test au seuil sélectionné": "{:.3f}",
        "Précision au seuil sélectionné": "{:.3f}",
        "Rappel au seuil sélectionné": "{:.3f}",
        "Brier score": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 9. Explication des métriques
# --------------------------------------------------

with st.expander("Comment lire les métriques ?"):

    st.markdown("""
- **Précision** : parmi les messages prédits phishing, proportion
  réellement étiquetée phishing.
- **Rappel** : parmi les messages phishing, proportion détectée.
- **F1-score** : compromis entre précision et rappel.
- **Brier score** : erreur quadratique moyenne entre les scores et
  les labels; plus bas est généralement préférable.
- Le seuil est sélectionné sur la validation, pas sur le test.
  Un seuil élevé peut réduire les fausses alertes, mais aussi laisser
  passer davantage de messages de phishing.
""")

st.info(
    f"Seuil de décision actuel : {threshold:.2f}. "
    "Utilise le résultat pour prioriser une vérification humaine, "
    "sans bloquer automatiquement les messages."
)

st.caption(
    "Dataset : phishing-email-balanced-6000 — "
    "https://huggingface.co/datasets/jhonrayo99/"
    "phishing-email-balanced-6000"
)
