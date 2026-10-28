
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from data_utils import load_dataset


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """Calcule les métriques pour un seuil donné."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                probabilities
            )
        ),
        "threshold": float(threshold),
    }


def select_threshold(y_val, val_probabilities):
    """
    Sélectionne le seuil qui maximise le F1-score
    sur les données de validation uniquement.

    En cas d'égalité, préfère le seuil le plus proche de 0.50.
    """

    thresholds = np.round(
        np.arange(0.20, 0.801, 0.02),
        2
    )

    results = []

    for threshold in thresholds:
        result = calculate_metrics(
            y_val,
            val_probabilities,
            threshold
        )
        results.append(result)

    best = max(
        results,
        key=lambda result: (
            result["f1"],
            -abs(result["threshold"] - 0.50)
        )
    )

    return best


def main():

    # --------------------------------------------------
    # 1. Chargement du dataset
    # --------------------------------------------------

    df = load_dataset()

    print("\nNombre total de messages :", len(df))
    print("0 = légitime ; 1 = phishing")

    # --------------------------------------------------
    # 2. Séparation : 70 % / 15 % / 15 %
    # --------------------------------------------------

    X_train_all, X_test, y_train_all, y_test = (
        train_test_split(
            df["text"],
            df["label"],
            test_size=0.15,
            random_state=42,
            stratify=df["label"],
        )
    )

    # 15 % du dataset total pour la validation.
    # Après avoir retiré 15 % pour le test, cela correspond
    # à 0.15 / 0.85 de la partie restante.

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=y_train_all,
    )

    print("\nRépartition des données :")
    print(f"Entraînement : {len(X_train)} messages")
    print(f"Validation   : {len(X_val)} messages")
    print(f"Test         : {len(X_test)} messages")

    # --------------------------------------------------
    # 3. Configuration TF-IDF
    # --------------------------------------------------

    tfidf_options = {
        "ngram_range": (1, 2),
        "max_features": 20000,
        "min_df": 2,
        "lowercase": True,
        "strip_accents": "unicode",
        "sublinear_tf": True,
    }

    # --------------------------------------------------
    # 4. Création des modèles
    # --------------------------------------------------

    models = {
        "logistic": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**tfidf_options)
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ]),

        "naive_bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**tfidf_options)
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.3)
            ),
        ]),
    }

    report = {}

    # --------------------------------------------------
    # 5. Entraînement et évaluation
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\n{'=' * 50}")
        print("Entraînement :", name)

        # Apprendre uniquement sur les données d'entraînement.
        model.fit(X_train, y_train)

        # Scores de phishing sur validation.
        val_probabilities = model.predict_proba(X_val)[:, 1]

        # Choisir le seuil sur validation uniquement.
        best = select_threshold(
            y_val,
            val_probabilities
        )

        selected_threshold = best["threshold"]

        print(
            "Seuil sélectionné sur validation :",
            f"{selected_threshold:.2f}"
        )

        print(
            "F1 validation au seuil sélectionné :",
            f"{best['f1']:.3f}"
        )

        # Le test est évalué après la sélection du seuil.
        test_probabilities = model.predict_proba(X_test)[:, 1]

        # Résultats au seuil classique de 0.50.
        test_at_05 = calculate_metrics(
            y_test,
            test_probabilities,
            0.50
        )

        # Résultats au seuil sélectionné sur validation.
        test_at_selected = calculate_metrics(
            y_test,
            test_probabilities,
            selected_threshold
        )

        # Matrices de confusion du jeu de test.
        test_predictions_05 = (
            test_probabilities >= 0.50
        ).astype(int)

        test_predictions_selected = (
            test_probabilities >= selected_threshold
        ).astype(int)

        cm_at_05 = confusion_matrix(
            y_test,
            test_predictions_05,
            labels=[0, 1]
        )

        cm_at_selected = confusion_matrix(
            y_test,
            test_predictions_selected,
            labels=[0, 1]
        )

        report[name] = {
            # Compatibilité avec l'ancien rapport.
            "precision": test_at_05["precision"],
            "recall": test_at_05["recall"],
            "f1": test_at_05["f1"],
            "brier": test_at_05["brier"],

            "threshold_selected_on_validation": (
                selected_threshold
            ),

            "validation_metrics_at_selected_threshold": best,

            "test_metrics_at_0_5": test_at_05,

            "test_metrics_at_selected_threshold": (
                test_at_selected
            ),

            "test_confusion_matrix_at_0_5": (
                cm_at_05.tolist()
            ),

            "test_confusion_matrix_at_selected_threshold": (
                cm_at_selected.tolist()
            ),
        }

        print("\nMétriques test au seuil 0.50 :")
        print(test_at_05)

        print("\nMétriques test au seuil sélectionné :")
        print(test_at_selected)

    # --------------------------------------------------
    # 6. Préparer quelques exemples du test
    # --------------------------------------------------
    # Ces exemples servent à la démonstration dans l'interface.
    # Ils ne servent pas à calculer les métriques finales.

    logistic_probabilities = models[
        "logistic"
    ].predict_proba(X_test)[:, 1]

    test_df = pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "label": y_test.reset_index(drop=True),
    })

    # Utilisation du seuil 0.50 pour identifier les catégories
    # d'exemples de démonstration.
    test_df["pred"] = (
        logistic_probabilities >= 0.50
    ).astype(int)

    examples = []

    choices = [
        ("Phishing — détecté", 1, 1),
        ("Phishing — manqué", 1, 0),
        ("Légitime — bien classé", 0, 0),
        ("Légitime — fausse alerte", 0, 1),
    ]

    for title, true_label, predicted_label in choices:

        rows = test_df[
            (test_df["label"] == true_label)
            & (test_df["pred"] == predicted_label)
        ]

        if not rows.empty:
            row = rows.iloc[0]

            examples.append({
                "example_name": title,
                "text": row["text"],
                "label": int(row["label"]),
                "logistic_prediction": int(row["pred"]),
            })

    # Compléter si certaines catégories sont absentes.
    for true_label, title in [
        (1, "Phishing — exemple test"),
        (0, "Légitime — exemple test"),
    ]:

        rows = test_df[test_df["label"] == true_label]

        for _, row in rows.iterrows():

            if len(examples) >= 4:
                break

            if not any(
                item["text"] == row["text"]
                for item in examples
            ):
                examples.append({
                    "example_name": (
                        f"{title} {len(examples) + 1}"
                    ),
                    "text": row["text"],
                    "label": int(row["label"]),
                    "logistic_prediction": int(row["pred"]),
                })

        if len(examples) >= 4:
            break

    # Assurer des noms d'exemples uniques.
    used_names = set()

    for example in examples:

        base_name = example["example_name"]
        suffix = 2

        while example["example_name"] in used_names:
            example["example_name"] = (
                f"{base_name} ({suffix})"
            )
            suffix += 1

        used_names.add(example["example_name"])

    # --------------------------------------------------
    # 7. Sauvegarder les modèles et les résultats
    # --------------------------------------------------

    joblib.dump(
        models,
        ART / "models.joblib"
    )

    (ART / "metrics.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    pd.DataFrame(examples).to_csv(
        ART / "test_examples.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nModèles enregistrés dans :", ART)
    print("Fichier métriques :", ART / "metrics.json")
    print("Fichier exemples :", ART / "test_examples.csv")


if __name__ == "__main__":
    main()
