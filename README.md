# 🛰️ Disaster Tweets NLP Classification

### *Classification supervisée de tweets en situation de crise via des Transformers pré-entraînés*

[![CI Pipeline](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disaster-tweets-nlp-ml.streamlit.app/)

Classification binaire de tweets pour distinguer les véritables alertes de catastrophes (incendies, séismes, inondations) du langage figuré des réseaux sociaux, afin de permettre aux organismes de secours et aux agences de presse de filtrer le bruit en temps réel.

`Python` `BERT` `DistilBERT` `RoBERTa` `PyTorch` `MLflow` `FastAPI` `Streamlit` `Ruff` `Pytest` `GitHub Actions`

---

## 📑 Table des matières

1. [Contexte business](#contexte-business)
2. [Architecture du projet](#architecture-du-projet)
3. [Dataset](#dataset)
4. [Installation pas à pas](#installation-pas-à-pas)
5. [Description de chaque fichier](#description-de-chaque-fichier)
6. [Pipeline MLOps — ordre d'exécution](#pipeline-mlops--ordre-dexécution)
7. [Résultats](#résultats)
8. [Tests unitaires](#tests-unitaires)
9. [CI/CD GitHub Actions](#cicd-github-actions)
10. [API FastAPI](#api-fastapi)
11. [Dashboard Streamlit](#dashboard-streamlit)
12. [Auteurs](#auteurs)

---

## Contexte business

**Problème :** Twitter est devenu un canal de communication vital en cas d'urgence. L'omniprésence des smartphones permet aux gens d'annoncer une urgence en temps réel. Mais il n'est pas toujours évident de savoir si les mots d'une personne annoncent réellement une catastrophe. *"This song is absolute fire!"* n'a rien à voir avec un incendie, tandis que *"Wildfire approaching the city, evacuations ordered"* est une urgence réelle.

**Solution :** Ce projet développe un pipeline NLP complet, des baselines classiques (TF-IDF + SVM) jusqu'aux Transformers (BERT, DistilBERT, RoBERTa), pour identifier automatiquement les tweets décrivant de vraies catastrophes. Chaque expérience est tracée avec MLflow, le modèle champion est déployé via une API FastAPI sur Hugging Face Spaces, et une interface Streamlit permet à n'importe qui de tester le système.

**Métrique retenue : F1-Score (classe Disaster)**

Le F1-Score a été choisi car il pénalise aussi bien les alertes manquées (faux négatifs → des vies en danger) que les fausses alertes (faux positifs → saturation des secours). L'Accuracy serait trompeuse à cause du déséquilibre naturel des classes.

---

## Architecture du projet

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS                       │
│                                                         │
│  data/raw data/   →  notebooks/EDA/EDA.ipynb            │
│  (CSV Kaggle)        Exploration des données            │
│       ↓                                                 │
│  notebooks/       →  NB1–NB4 (Baselines classiques)     │
│  Modelisation/       TF-IDF, Char n-grams, Embeddings   │
│       ↓                                                 │
│  notebooks/       →  NB5–NB7 (Transformers)             │
│  Modelisation/       DistilBERT, BERT-base, RoBERTa     │
│       ↓                                                 │
│  notebooks/       →  NB9 (Sélection finale)             │
│  Modelisation/       Comparaison MLflow de tous les runs │
│       ↓                                                 │
│  models/          →  distilbert_disaster/               │
│  artefacts           Poids + tokenizer exportés          │
│       ↓                                                 │
│  api.py           →  FastAPI sur Hugging Face Spaces     │
│                      POST /predict                      │
│       ↓                                                 │
│  web application/ →  app.py (Streamlit Cloud)           │
│                      Interface Premium Glassmorphism     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUALITÉ & CI/CD                      │
│                                                         │
│  tests/           →  4 tests unitaires Pytest           │
│  test_api.py         Mock du modèle, validation API     │
│       ↓                                                 │
│  .github/         →  GitHub Actions                     │
│  workflows/ci.yml    Ruff + Pytest à chaque git push    │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

Le projet utilise le dataset Kaggle **"Natural Language Processing with Disaster Tweets"**.

| Fichier | Lignes | Colonnes | Rôle |
| :--- | :---: | :---: | :--- |
| `train.csv` | 7 613 | 5 | Tweets annotés (target 0/1) |
| `test.csv` | 3 263 | 4 | Tweets non annotés (soumission) |

**Colonnes disponibles :**
- `text` : contenu du tweet (seule feature utilisée par notre modèle)
- `keyword` : mot-clé associé (optionnel, souvent vide)
- `location` : localisation déclarée (optionnel, très bruitée)
- `target` : 0 = pas une catastrophe, 1 = catastrophe réelle

**Statistiques clés :**
- 83% des tweets sont de classe 0 (pas une catastrophe)
- 17% des tweets sont de classe 1 (catastrophe réelle)
- Le déséquilibre est géré par **stratification** lors du split train/validation

---

## Installation pas à pas

### Prérequis
```bash
python --version   # >= 3.10 recommandé
git --version      # >= 2.0
```

### Étape 1 — Cloner le repository
```bash
git clone https://github.com/ahmadouniass/Disaster-Tweets-NLP.git
cd Disaster-Tweets-NLP
```

### Étape 2 — Créer l'environnement virtuel
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 — Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 — Télécharger le dataset
Téléchargez depuis Kaggle : https://www.kaggle.com/competitions/nlp-getting-started/data

Placez les CSV dans `data/raw data/`.

### Étape 5 — Lancer l'API (en local)
```bash
python api.py
# Documentation interactive : http://localhost:8000/docs
```

### Étape 6 — Lancer le dashboard
```bash
streamlit run "web application/app.py"
```

---

## Description de chaque fichier

### Notebooks — Exploration (`notebooks/EDA/`)

#### `EDA.ipynb`
**Rôle :** Analyse exploratoire des données (EDA).

Ce notebook explore le dataset Kaggle pour comprendre la structure des données avant toute modélisation. Il analyse la distribution des classes, la longueur des tweets, les mots-clés les plus fréquents par classe, et les valeurs manquantes dans `keyword` et `location`.

Ce qu'il produit :
- Distribution des classes (43% disaster vs 57% non-disaster)
- Nuages de mots par classe
- Analyse de la longueur des tweets
- Décision finale : utiliser uniquement la colonne `text`

### Notebooks — Modélisation (`notebooks/Modelisation/`)

#### `NB1_count_tfidf_baselines_mlflow_tuning_in_notebook.ipynb`
**Rôle :** Baselines avec CountVectorizer et TF-IDF.

Ce notebook établit les performances de référence avec des pipelines sklearn classiques : Naive Bayes, Logistic Regression, SVM, combinés avec CountVectorizer et TF-IDF. Chaque combinaison est enregistrée dans MLflow. Ces baselines servent de point de comparaison pour évaluer le gain apporté par les Transformers.

#### `NB2_weighting_char_hybrid_mlflow_tuning_in_notebook.ipynb`
**Rôle :** Variantes avancées de vectorisation textuelle.

Ce notebook explore des stratégies de pondération alternatives (Sublinear TF, BM25) et des **Character n-grams** pour capturer la morphologie des mots. L'objectif est de vérifier si des représentations textuelles plus fines améliorent les baselines.

#### `NB3_reduction_selection_nb_mlflow_tuning_in_notebook.ipynb`
**Rôle :** Réduction de dimensionnalité et sélection de features.

Ce notebook teste l'effet de la réduction de dimensionnalité (TruncatedSVD, SelectKBest) sur les performances des pipelines TF-IDF. Il vérifie si réduire le bruit dimensionnel améliore la généralisation.

#### `NB4_classical_sentence_embeddings_mlflow_tuning_in_notebook.ipynb`
**Rôle :** Sentence Embeddings pré-entraînés.

Ce notebook utilise des embeddings denses (GloVe Twitter 200d via Gensim, Sentence-Transformers MiniLM) pour remplacer TF-IDF par des vecteurs sémantiques. C'est la transition entre l'approche fréquentielle et l'approche contextuelle.

#### `NB5_DistilBERT_finetuning_baseline_tuning.ipynb`
**Rôle :** Fine-tuning de DistilBERT.

Premier notebook Transformer. Il charge `distilbert-base-uncased` depuis Hugging Face, le fine-tune sur le dataset de tweets avec le Trainer API, et évalue les performances. DistilBERT est 60% plus rapide que BERT-base, ce qui en fait le candidat idéal pour le déploiement en production.

#### `NB6_BERT_base_uncased_finetuning_baseline_tuning.ipynb`
**Rôle :** Fine-tuning de BERT-base-uncased — **Modèle Champion**.

Ce notebook entraîne le modèle de référence de Google avec un scheduler de learning rate linéaire, un batch size de 8, et 3 époques. Il teste également plusieurs configurations d'hyperparamètres (learning rate, batch size, epochs) dans une phase de tuning léger. C'est ici que nous obtenons le **meilleur F1-Score (78.0%)**.

Configuration retenue :
- `learning_rate` : 2e-05
- `batch_size` : 8
- `num_epochs` : 3
- `weight_decay` : 0.01
- `max_len` : 96

#### `NB7_RoBERTa_base_finetuning_baseline_tuning.ipynb`
**Rôle :** Fine-tuning de RoBERTa-base.

Test d'une variante plus robuste de BERT, entraînée sur plus de données et sans le Next Sentence Prediction. L'objectif est de vérifier si ce surplus de pré-entraînement apporte un gain significatif sur notre tâche spécifique.

#### `NB9_Model_Comparison_and_Final_Selection.ipynb`
**Rôle :** Synthèse et sélection finale.

Ce notebook charge tous les résultats des notebooks précédents et produit un classement final basé sur le F1-Score de la classe Disaster. Il justifie le choix de BERT-base comme champion scientifique et de DistilBERT comme modèle de production.

#### `api.py`
**Rôle :** API REST FastAPI exposant le modèle DistilBERT pour l'inférence.

L'API charge automatiquement le modèle et le tokenizer depuis `models/distilbert_disaster/` au démarrage. Elle expose 2 endpoints :

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/` | Health check — l'API est-elle opérationnelle ? |
| POST | `/predict` | Prédit si un tweet décrit une catastrophe |

**Logique d'inférence :**
1. Le texte est tokenisé avec `AutoTokenizer`
2. Les tokens passent dans le modèle DistilBERT (`model.eval()`, `torch.no_grad()`)
3. Les logits sont convertis en probabilités via `F.softmax`
4. La classe avec la probabilité maximale est retournée avec le score de confiance

Corps de la requête POST `/predict` :
```json
{"text": "Huge wildfire approaching the city!"}
```
Réponse :
```json
{"prediction": "Disaster", "confidence": 0.97}
```

**Déployé sur :** https://ahmedtrip-disaster-tweet-api.hf.space/docs

#### `select_best_model.py`
**Rôle :** Script standalone de comparaison des modèles.

Ce script scanne le dossier `outputs/` pour trouver tous les fichiers de résultats (`*_results.csv`), les fusionne en un classement unique trié par F1-Score, et affiche le podium des meilleurs modèles. Utile pour une vue d'ensemble rapide sans ouvrir MLflow.

```bash
python select_best_model.py
# 🏆 LE CHAMPION EST : BERT_base_uncased
# 📈 F1-Score : 0.7803
```

### Interface utilisateur (`web application/`)

#### `app.py`
**Rôle :** Dashboard Streamlit Premium avec design Glassmorphism.

Cette application web embarque une interface HTML/CSS/JS complète dans un composant Streamlit. Elle communique avec l'API Hugging Face pour les prédictions.

Fonctionnalités :
- **Mode Sombre/Clair** : toggle intégré dans la barre de navigation
- **Navigation sticky** : barre fixe avec défilement fluide vers Home, Mission, Analyze, Team
- **Exemples cliquables** : 4 tweets pré-remplis pour tester rapidement le modèle
- **Résultats visuels** : cartes animées avec barre de confiance colorée (rouge = disaster, vert = safe)
- **Galerie de l'équipe** : avatars GitHub cliquables avec liens vers les profils

**Déployé sur :** https://disaster-tweets-nlp-ml.streamlit.app/


#### `LICENSE`
Licence MIT — usage libre avec attribution.

### Tests (`tests/`)

#### `test_api.py`
**Rôle :** tests unitaires pour l'API FastAPI.

**Technique :** Les tests utilisent `MagicMock` pour simuler le modèle DistilBERT et le tokenizer sans charger les poids réels (260 Mo). `mock_outputs.logits = torch.tensor([[0.1, 0.9]])` simule une prédiction "Disaster" avec haute confiance.

### Données et modèles

#### `data/raw data/`
Contient les CSV bruts du dataset Kaggle déjà splitted(`train.csv`, `test.csv`).

#### `data/processed data/`
Contient les données nettoyées et préparées pour l'entraînement.

#### `models/distilbert_disaster/`
Contient les artefacts du modèle de production : poids du réseau (`pytorch_model.bin` ou `model.safetensors`), configuration (`config.json`), vocabulaire du tokenizer (`vocab.txt`, `tokenizer.json`).

#### `outputs/`
Dépôt central des résultats. Organisé par notebook (`NB1/`, `NB2/`, ..., `NB6/`, `NB8/`). Chaque sous-dossier contient les CSV de métriques par pipeline et le dossier `mlruns/` contient la base de données locale MLflow.

---

## Pipeline MLOps — ordre d'exécution

```bash
# 1. Cloner et installer
git clone https://github.com/ahmadouniass/Disaster-Tweets-NLP.git
cd Disaster-Tweets-NLP
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Télécharger les données depuis Kaggle
# Placer train.csv et test.csv dans data/raw data/

# 3. Exploration (optionnel)
jupyter notebook notebooks/EDA/EDA.ipynb

# 4. Baselines classiques (NB1 → NB4)
jupyter notebook "notebooks/Modelisation/NB1_count_tfidf_baselines_mlflow_tuning_in_notebook (1).ipynb"

# 5. Fine-tuning Transformers (NB5 → NB7) — nécessite GPU (Google Colab recommandé)
jupyter notebook "notebooks/Modelisation/NB6_BERT_base_uncased_finetuning_baseline_tuning (2).ipynb"

# 6. Sélection du champion
jupyter notebook notebooks/Modelisation/NB9_Model_Comparison_and_Final_Selection.ipynb
# ou
python select_best_model.py

# 7. Lancer les tests
python -m pytest tests/ -v

# 8. Vérifier la qualité du code
ruff check .

# 9. Lancer l'API
python api.py
# → http://localhost:8000/docs

# 10. Lancer le dashboard
streamlit run "web application/app.py"

# 11. Visualiser les expériences MLflow (optionnel)
mlflow ui --backend-store-uri outputs/mlruns --port 5000
# → http://localhost:5000

# 12. Commiter et pusher (déclenche CI/CD)
git add .
git commit -m "feat: pipeline complet"
git push   # → GitHub Actions exécute Ruff + Pytest
```

---

## Résultats

### Comparaison des approches

| Approche | Modèle | F1-Score (Disaster) | Accuracy | Rôle |
| :--- | :--- | :---: | :---: | :--- |
| Fréquentielle | TF-IDF + SVM | ~71% | ~79% | Baseline |
| Embeddings | GloVe + LR | ~73% | ~80% | Intermédiaire |
| **Transformer** | **BERT-base** | **78.0%** | **91.3%** | **Champion** |
| **Transformer** | **DistilBERT** | **77.3%** | **90.8%** | **Production** |
| Transformer | RoBERTa | 76.9% | 90.5% | Challenger |

---



## CI/CD GitHub Actions

Le pipeline `.github/workflows/ci.yml` s'exécute automatiquement à chaque `git push` sur les branches `main`, `fix/deployment` et `fix/ci-linting`.

### Étapes du pipeline

```
1. actions/checkout@v4          Récupère le code
2. actions/setup-python@v5      Installe Python 3.10
3. pip install -r requirements  Installe les dépendances
4. ruff check .                 Vérifie la qualité du code (PEP8, imports)
5. python -m pytest             Exécute les  tests unitaires
```

---

## API FastAPI

**Production :** https://ahmedtrip-disaster-tweet-api.hf.space

**Documentation Swagger :** https://ahmedtrip-disaster-tweet-api.hf.space/docs

### Endpoints

**GET /** — Health check
```json
{"message": "Disaster Tweet Prediction API is running"}
```

**POST /predict** — Prédiction
```json
// Requête
{"text": "Serious flooding reported downtown after the storm."}

// Réponse
{"prediction": "Disaster", "confidence": 0.94}
```

La confiance provient de la fonction **Softmax** appliquée aux logits bruts du modèle. Elle représente la probabilité que le tweet appartienne à la classe prédite.

---

## Dashboard Streamlit

**Production :** https://disaster-tweets-nlp-ml.streamlit.app/

---

## Stack technique

| Catégorie | Technologie | Rôle |
| :--- | :--- | :--- |
| Langage | Python 3.10 | Langage principal |
| Deep Learning | PyTorch + Transformers | Fine-tuning BERT/DistilBERT/RoBERTa |
| ML classique | scikit-learn | Baselines TF-IDF + SVM/NB/LR |
| Tracking | MLflow | Logging paramètres, métriques, artefacts |
| API | FastAPI + Uvicorn | Endpoint de prédiction REST |
| Interface | Streamlit | Dashboard marketing Premium |
| Tests | Pytest | 4 tests unitaires avec Mock |
| Qualité | Ruff | Linter/Formatter PEP8 |
| CI/CD | GitHub Actions | Pipeline automatisé |
| Hébergement | Hugging Face Spaces | API de production |
| Hébergement | Streamlit Cloud | Dashboard de production |

---

## Équipe & Auteurs

| <div align="center"><img src="https://github.com/ahmadouniass.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/Khadidiatou1010.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/dior204.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/Kerencia2.png" width="100" style="border-radius:50%"></div> |
| :---: | :---: | :---: | :---: |
| [**Ahmadou Niass**](https://github.com/ahmadouniass) | [**Khadidiatou Coulibaly**](https://github.com/Khadidiatou1010) | [**Dior Mbengue**](https://github.com/dior204) | [**Pahane S. K. D.**](https://github.com/Kerencia2) |

Projet réalisé dans le cadre du cours Machine Learning 2 — ENSAE Dakar, 2026.

---
<p align="center">Developed with 💙 by the Team</p>
