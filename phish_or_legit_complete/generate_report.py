
from pathlib import Path
import html
import json

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import cm


ROOT = Path(__file__).resolve().parent

ART = ROOT / "artifacts"

OUT = ROOT / "reports" / "rapport_phish_or_legit.pdf"


def main():

    metrics_file = ART / "metrics.json"
    examples_file = ART / "test_examples.csv"

    if not metrics_file.exists() or not examples_file.exists():
        raise FileNotFoundError(
            "Les résultats sont introuvables. "
            "Exécute d'abord : python train_model.py"
        )

    results = json.loads(
        metrics_file.read_text(encoding="utf-8")
    )

    examples = pd.read_csv(examples_file)

    # --------------------------------------------------
    # 1. Styles du document
    # --------------------------------------------------

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
        )
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1 * cm,
        bottomMargin=1 * cm,
    )

    story = []

    # --------------------------------------------------
    # 2. Introduction
    # --------------------------------------------------

    story.extend([
        Paragraph(
            "Phish or Legit? — Rapport d'évaluation",
            styles["Title"],
        ),

        Spacer(1, 8),

        Paragraph(
            "Objectif : classifier des emails en légitimes ou phishing. "
            "Le texte est converti en caractéristiques TF-IDF "
            "(unigrammes et bigrammes). Deux modèles sont comparés : "
            "régression logistique et Naive Bayes multinomial. "
            "Le dataset public est nettoyé, puis séparé en entraînement, "
            "validation et test. Le seuil est choisi sur validation, "
            "puis évalué sur un jeu de test indépendant.",
            styles["SmallText"],
        ),

        Spacer(1, 10),
    ])

    # --------------------------------------------------
    # 3. Métriques au seuil 0.50
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Résultats sur le test au seuil 0,50",
            styles["Heading2"],
        )
    )

    table_data = [
        [
            "Modèle",
            "Précision",
            "Rappel",
            "F1-score",
            "Brier",
        ]
    ]

    model_labels = [
        ("logistic", "Régression logistique"),
        ("naive_bayes", "Naive Bayes multinomial"),
    ]

    for key, label in model_labels:

        metrics = results[key]["test_metrics_at_0_5"]

        table_data.append([
            label,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['brier']:.3f}",
        ])

    table = Table(
        table_data,
        colWidths=[
            6 * cm,
            2.2 * cm,
            2 * cm,
            1.5 * cm,
            1.5 * cm,
        ],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#24415a"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
        ])
    )

    story.extend([
        table,
        Spacer(1, 10),
    ])

    # --------------------------------------------------
    # 4. Métriques au seuil choisi sur validation
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Résultats au seuil sélectionné sur validation",
            styles["Heading2"],
        )
    )

    selected_data = [
        [
            "Modèle",
            "Seuil",
            "Précision",
            "Rappel",
            "F1-score",
        ]
    ]

    for key, label in model_labels:

        model_results = results[key]

        selected_metrics = model_results[
            "test_metrics_at_selected_threshold"
        ]

        selected_data.append([
            label,
            (
                f"{model_results['threshold_selected_on_validation']:.2f}"
            ),
            f"{selected_metrics['precision']:.3f}",
            f"{selected_metrics['recall']:.3f}",
            f"{selected_metrics['f1']:.3f}",
        ])

    selected_table = Table(
        selected_data,
        colWidths=[
            6 * cm,
            2 * cm,
            2.2 * cm,
            2 * cm,
            1.5 * cm,
        ],
    )

    selected_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#24415a"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.grey,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
        ])
    )

    story.extend([
        selected_table,
        Spacer(1, 10),
    ])

    # --------------------------------------------------
    # 5. Explication du seuil
    # --------------------------------------------------

    logistic_threshold = results[
        "logistic"
    ]["threshold_selected_on_validation"]

    story.append(
        Paragraph(
            "Seuil et politique de décision",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Seuil de régression logistique sélectionné sur validation : "
            f"{logistic_threshold:.2f}. Le seuil est choisi en maximisant "
            f"le F1-score sur les données de validation. Un seuil plus "
            f"élevé peut réduire les fausses alertes, mais risque aussi "
            f"de laisser passer davantage de phishing. La décision doit "
            f"servir à prioriser une vérification humaine et non à bloquer "
            f"automatiquement les messages.",
            styles["SmallText"],
        )
    )

    story.append(Spacer(1, 10))

    # --------------------------------------------------
    # 6. Exemples du jeu de test
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Exemples issus du jeu de test",
            styles["Heading2"],
        )
    )

    for _, row in examples.head(4).iterrows():

        name = html.escape(str(row["example_name"]))
        message = html.escape(str(row["text"]))

        actual_label = (
            "phishing"
            if int(row["label"]) == 1
            else "légitime"
        )

        story.append(
            Paragraph(
                f"<b>{name}</b> — label réel : {actual_label}.",
                styles["SmallText"],
            )
        )

        truncated_message = (
            message[:350] + ("..." if len(message) > 350 else "")
        )

        story.append(
            Paragraph(
                truncated_message,
                styles["SmallText"],
            )
        )

        story.append(Spacer(1, 4))

    # --------------------------------------------------
    # 7. Limites
    # --------------------------------------------------

    story.extend([
        Spacer(1, 8),

        Paragraph(
            "Limites de l'étude",
            styles["Heading2"],
        ),

        Paragraph(
            "Le corpus est en anglais et équilibré; il peut donc différer "
            "de la distribution réelle d'une boîte mail. Les labels "
            "peuvent être imparfaits. Les scores du modèle ne garantissent "
            "pas la détection de toutes les attaques. Une vérification "
            "humaine reste nécessaire.",
            styles["SmallText"],
        ),
    ])

    # --------------------------------------------------
    # 8. Générer le PDF
    # --------------------------------------------------

    doc.build(story)

    print("Rapport PDF créé :", OUT)


if __name__ == "__main__":
    main()
