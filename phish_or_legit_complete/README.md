# Phish or Legit? — projet complet

Application Streamlit de classification d'emails anglais : `0 = safe/légitime`, `1 = phishing`. Elle compare TF-IDF + régression logistique et TF-IDF + Naive Bayes multinomial.

Dataset : [jhonrayo99/phishing-email-balanced-6000](https://huggingface.co/datasets/jhonrayo99/phishing-email-balanced-6000), 6 000 emails anglais (3 000 par classe), colonnes `text` et `label`, licence déclarée LGPL-3.0. Le téléchargement est automatique au premier lancement.

## Installer sous Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si l'activation PowerShell est bloquée, utilise `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Entraîner et évaluer

```powershell
python train_model.py
```

Le script nettoie les données, effectue une séparation stratifiée train/validation/test, ajuste le vectoriseur uniquement sur le train, choisit le seuil sur validation et calcule précision, rappel, F1 et Brier sur le test. Les artefacts sont enregistrés dans `artifacts/`.

## Lancer l'application

```powershell
streamlit run app.py
```

L'application comprend la saisie de message, le choix du modèle, le seuil dynamique, des exemples du jeu de test, les contributions des caractéristiques et les métriques.

## Générer le rapport PDF

Après l'entraînement :

```powershell
python generate_report.py
```

Le PDF est généré dans `reports/rapport_phish_or_legit.pdf`. Relis-le et complète ton analyse avant de le remettre.

## Déploiement Streamlit Community Cloud

1. Envoie les fichiers du projet dans un dépôt GitHub.
2. Ne publie pas le dataset local ni les modèles entraînés (ils sont exclus par `.gitignore`).
3. Sur Streamlit Community Cloud, connecte le dépôt et sélectionne `app.py`.
4. Attends le premier téléchargement/entraînement et teste le lien public.

## Interprétation

- **Précision** : parmi les messages prédits phishing, part étiquetée phishing.
- **Rappel** : part des phishing étiquetés retrouvée.
- **F1** : compromis précision/rappel.
- **Brier score** : erreur quadratique du score; plus bas est préférable. Naive Bayes peut être mal calibré.
- Pour la régression logistique, contribution ≈ TF-IDF × coefficient. Pour Naive Bayes, la contribution affichée est une approximation du rapport des log-probabilités.
- Le dataset est équilibré; il ne reflète pas nécessairement la fréquence du phishing dans une vraie boîte mail.
- Ne clique pas sur les liens de messages suspects. L'application n'effectue aucun blocage automatique.

## Important par rapport au guide du TP

Le guide fourni décrit un dataset synthétique de 2 000 messages. Cette version utilise un corpus public réel de 6 000 emails; les valeurs attendues dans le guide ne s'appliqueront donc pas. Demande l'accord de l'enseignant si le dataset fourni est obligatoire.
