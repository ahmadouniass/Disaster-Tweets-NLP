# 🛰️ Disaster Tweets NLP Classification

[![CI Pipeline](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disaster-tweets-nlp-ml.streamlit.app/)

## 🔗 Liens Utiles
- **🌐 Application Web :** [disaster-tweets-nlp-ml.streamlit.app](https://disaster-tweets-nlp-ml.streamlit.app/)
- **⚙️ API Backend :** [ahmedtrip-disaster-tweet-api.hf.space/docs](https://ahmedtrip-disaster-tweet-api.hf.space/docs)

## 📝 Présentation du Projet
Ce projet s'attaque au défi de la classification de tweets effectués en situation de crise. L'enjeu est de filtrer le bruit des réseaux sociaux pour identifier les alertes réelles de catastrophes (incendies, inondations, séismes) parmi les messages utilisant un langage figuré ou sans danger. 

Le système repose sur un pipeline de Deep Learning de pointe, intégré dans une architecture logicielle moderne incluant une API de service et une interface utilisateur premium.

---

## 🔍 Analyse et Traitement des Données (EDA)
Le dataset (Kaggle NLP Getting Started) a fait l'objet d'une analyse exploratoire approfondie :
- **Nettoyage Textuel :** Suppression des caractères spéciaux, normalisation des emojis, gestion des URLs et des mentions.
- **Analyse Statistique :** Étude de la distribution des classes (43% Disaster), analyse de la longueur des tweets.
- **Feature Engineering :** Extraction de caractéristiques textuelles pour enrichir les modèles classiques.

## 🧠 Modélisation et Expérimentation
Nous avons exploré le spectre complet de l'état de l'art en NLP. Chaque expérience a été tracée avec **MLflow**.

### 🏆 Résultats du Modèle Champion : BERT-base-uncased (Fine-tuned)
Le **F1-Score sur la classe Disaster (1)** a été retenu comme métrique principale. Ce choix permet de maximiser la détection des alertes réelles (Rappel) tout en minimisant les fausses alertes (Précision), ce qui est crucial pour la fiabilité d'un système de veille de crise.

| Métrique | Valeur (Test Set) |
| :--- | :--- |
| **F1-Score (Catastrophe)** | **78.03 %** |
| **Accuracy Globale** | **91.38 %** |
| **ROC AUC** | **95.12 %** |

#### Pourquoi ce choix ? (Analyse Comparative)
Nos expérimentations ont montré que certains modèles (comme **P16 - GloVe Twitter**) atteignent un Rappel supérieur (**86.05 %**). Cependant, ces modèles souffrent d'une Précision très faible (**54.49 %**), générant ainsi une fausse alerte sur deux.

En privilégiant le **F1-Score**, nous avons sélectionné **BERT-base** (et sa version distillée) qui offre le meilleur compromis : une détection robuste des catastrophes réelles tout en maintenant une fiabilité élevée.

> [!NOTE]
> Bien que **BERT-base** soit notre champion scientifique (meilleur score), nous avons maintenu **DistilBERT** pour le déploiement de l'API. Ce choix stratégique permet d'offrir une réponse quasi-instantanée aux utilisateurs tout en conservant 99% de la précision du modèle champion.

## 🏗️ Structure du Repository
- `api.py` : Serveur d'inférence haute performance.
- `web application/` : Interface utilisateur Streamlit Premium.
- `notebooks/` :
    - `EDA/` : Analyse et visualisation des données.
    - `Modelisation/` : Pipelines d'entraînement (NB1 à NB9).
- `outputs/` : Dépôt central des résultats (CSV, métriques).
- `tests/` : Tests unitaires automatisés.

## 🛠️ Stack Technique
- **ML :** `PyTorch`, `Transformers`, `MLflow`.
- **Backend :** `FastAPI`.
- **Frontend :** `Streamlit`.
- **Infrastructures :** `Hugging Face Spaces` (API), `Streamlit Cloud` (Web App).

---

## 👥 Équipe du Projet
- **Ahmadou Niass** ([@ahmadouniass](https://github.com/ahmadouniass))
- **Khadidiatou Coulibaly** ([@Khadidiatou1010](https://github.com/Khadidiatou1010))
- **Dior Mbengue** ([@dior204](https://github.com/dior204))
- **Pahane Seunkam Kerencia Dyvana** ([@Kerencia2](https://github.com/Kerencia2))
