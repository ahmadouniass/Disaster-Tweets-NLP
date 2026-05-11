# 🛰️ Disaster Tweets NLP Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disaster-tweets-nlp-ml.streamlit.app/)
[![CI Pipeline](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions)

## 🔗 Liens Utiles
- **🌐 Application Web :** [disaster-tweets-nlp-ml.streamlit.app](https://disaster-tweets-nlp-ml.streamlit.app/)
- **⚙️ API Backend :** [ahmedtrip-disaster-tweet-api.hf.space/docs](https://ahmedtrip-disaster-tweet-api.hf.space/docs)

## Présentation du Projet
Ce projet s'attaque au défi de la classification de tweets effectués en situation de crise. L'enjeu est de filtrer le bruit des réseaux sociaux pour identifier les alertes réelles de catastrophes (incendies, inondations, séismes) parmi les messages utilisant un langage figuré ou sans danger. 

Le système repose sur un pipeline de Deep Learning, intégré dans une architecture logicielle moderne incluant une API de service et une interface utilisateur.

---

## Analyse et Traitement des Données (EDA)
Le dataset (Kaggle NLP Getting Started) a fait l'objet d'une analyse exploratoire approfondie :
- **Nettoyage Textuel :** Suppression des caractères spéciaux, normalisation des emojis, gestion des URLs et des mentions, et nettoyage des abréviations spécifiques aux réseaux sociaux.
- **Analyse Statistique :** Étude de la distribution des classes (légèrement déséquilibrée), analyse de la longueur des tweets et du Type-Token Ratio (TTR).
- **Feature Engineering :** Extraction de caractéristiques textuelles (nombre de mots, majuscules, ponctuation, présence de liens) pour enrichir les modèles classiques.

##  Modélisation et Expérimentation
Chaque expérience a été tracée avec **MLflow** pour garantir la reproductibilité.

### Architectures testées :
| Famille de Modèle | Description |
| :--- | :--- |
| **Baselines** | TF-IDF (Unigrammes/Bigrammes) + Logistic Regression & Naive Bayes. |
| **Optimisation** | Réduction de dimension via SVD (LSA) et sélection de features avec ANOVA. |
| **Embeddings** | Vecteurs denses via Word2Vec, FastText et Sentence-Transformers. |
| **Transformers** | Fine-tuning de modèles pré-entraînés (**DistilBERT**, **BERT**, **RoBERTa**). |

### 🏆 Résultats du Modèle Champion : DistilBERT Tuned
Le **F1-Score sur la classe Disaster (1)** a été retenu comme métrique principale. Ce choix permet de maximiser la détection des alertes réelles (Rappel) tout en minimisant les fausses alertes (Précision), ce qui est crucial pour la fiabilité d'un système de veille de crise.

| Métrique | Valeur (Test Set) |
| :--- | :--- |
| **F1-Score (Catastrophe)** | **77.26 %** |
| **Précision** | **74.78 %** |
| **Rappel (Recall)** | **79.91 %** |
| **ROC AUC** | **94.87 %** |

#### Pourquoi ce choix ? (Analyse Comparative)
Nos expérimentations ont montré que certains modèles (comme **P16 - GloVe Twitter**) atteignent un Rappel supérieur (**86.05 %**). Cependant, ces modèles souffrent d'une Précision très faible (**54.49 %**), générant ainsi une fausse alerte sur deux.

En privilégiant le **F1-Score**, nous avons sélectionné **DistilBERT** qui offre le meilleur compromis : il détecte **80 %** des catastrophes réelles tout en maintenant une fiabilité de **75 %**. C'est le modèle le plus équilibré pour une mise en production.

---

## Stack Technique
- **Machine Learning :** `PyTorch`, `Transformers (Hugging Face)`, `Scikit-learn`.
- **Tracking :** `MLflow`.
- **Backend :** `FastAPI`, `Uvicorn`, `Pydantic`.
- **Frontend :** `Streamlit` (Custom HTML/CSS Injector).
- **Qualité de code :** `Ruff` (Linter), `Pytest` (Unit Tests avec Mocking).
- **Infrastructures :** `Hugging Face Spaces` (API), `Streamlit Cloud` (Web App).

## Structure du Repository
- `api.py` : Pour la gestion de l'api haute performance.
- `web application/` : Interface utilisateur avec design premium et feedback en temps réel.
- `notebooks/` :
    - `EDA/` : Analyse et visualisation des données.
    - `models_training/` : Pipelines d'entraînement du modèle NB1 au NB9.
- `outputs/` : Historique complet des métriques, matrices de confusion et rapports CSV.
- `tests/` : Suite de tests automatisés validant la logique de l'API.

## Installation Locale
```bash
git clone https://github.com/ahmadouniass/Disaster-Tweets-NLP.git
cd Disaster-Tweets-NLP
pip install -r requirements.txt
python api.py
streamlit run "web application/app.py"
```

---

## 👥 Équipe du Projet
- **Ahmadou Niass** ([@ahmadouniass](https://github.com/ahmadouniass))
- **Khadidiatou Coulibaly** ([@Khadidiatou1010](https://github.com/Khadidiatou1010))
- **Dior Mbengue** ([@dior204](https://github.com/dior204))
- **Pahane Seunkam Kerencia Dyvana** ([@Kerencia2](https://github.com/Kerencia2))
