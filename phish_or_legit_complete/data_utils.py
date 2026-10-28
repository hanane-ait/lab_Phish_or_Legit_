
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "phishing_email_subset.csv"

DATA_URL = (
    "https://huggingface.co/datasets/jhonrayo99/"
    "phishing-email-balanced-6000/resolve/main/"
    "phishing_email_subset.csv"
)


def load_dataset():
    """Charge et nettoie le dataset de classification des emails."""

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print("Téléchargement du dataset public Hugging Face...")

        try:
            df_downloaded = pd.read_csv(DATA_URL)
            df_downloaded.to_csv(DATA_FILE, index=False)
            print("Dataset téléchargé :", DATA_FILE)

        except Exception as exc:
            raise RuntimeError(
                "Impossible de télécharger le dataset. "
                "Vérifie ta connexion Internet et l'URL Hugging Face."
            ) from exc

    df = pd.read_csv(DATA_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Les colonnes 'text' et 'label' sont requises. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    # Conserver uniquement les colonnes nécessaires.
    df = df[["text", "label"]].copy()

    # Nettoyer les textes.
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Convertir les labels en nombres.
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Supprimer les labels manquants.
    df = df.dropna(subset=["label"])

    # Garder uniquement les classes 0 et 1 et les textes non vides.
    df = df[
        (df["text"].str.len() > 0)
        & df["label"].isin([0, 1])
    ].copy()

    df["label"] = df["label"].astype(int)

    # Supprimer les doublons exacts texte-label.
    df = df.drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Le dataset est vide après le nettoyage."
        )

    if df["label"].nunique() != 2:
        raise ValueError(
            "Le dataset doit contenir les deux classes : "
            "0 = légitime et 1 = phishing."
        )

    print("Dataset chargé :", len(df), "messages")
    print("\nRépartition des classes :")
    print(df["label"].value_counts().sort_index())

    return df
