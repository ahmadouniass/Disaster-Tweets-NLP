# 🛰️ Disaster Tweets NLP Classification

[![CI Pipeline](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadouniass/Disaster-Tweets-NLP/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://disaster-tweets-nlp-ml.streamlit.app/)

## Liens Utiles
- [**🌐 Application Web :**](https://disaster-tweets-nlp-ml.streamlit.app/)
- [**⚙️ API Backend :**](https://ahmedtrip-disaster-tweet-api.hf.space/docs)
- [**⚙️ Présentation Canvas **](https://canva.link/i6bfyuysalp7vnq)

## Présentation du Projet
Ce projet s'attaque au défi de la classification de tweets effectués en situation de crise. L'enjeu est de identifier les alertes réelles de catastrophes (incendies, inondations, séismes) parmi le bruit des réseaux sociaux.

---

## Modélisation et Expérimentation
Nous avons exploré le spectre complet de l'état de l'art en NLP. Chaque expérience a été tracée avec **MLflow**.

### Résultats du Modèle Champion : BERT-base-uncased (Fine-tuned)
Le **F1-Score sur la classe Disaster (1)** a été retenu comme métrique principale.

| Métrique | Valeur (Test Set) |
| :--- | :--- |
| **F1-Score (Catastrophe)** | **78.03 %** |
| **Accuracy Globale** | **91.38 %** |
| **ROC AUC** | **95.12 %** |

#### Pourquoi ce choix ? (Analyse Comparative)
En privilégiant le **F1-Score**, nous avons sélectionné **BERT-base** (et sa version distillée) qui offre le meilleur compromis entre Rappel (détection) et Précision (fiabilité).

---

##  Structure du Repository
- `api.py` : Serveur d'inférence (FastAPI).
- `web application/` : Interface utilisateur Premium (Streamlit).
- `notebooks/` : Pipelines d'analyse et d'entraînement.
- `outputs/` : Dépôt central des résultats et métriques.

## Stack Technique
- **ML :** `PyTorch`, `Transformers`, `MLflow`.
- **Backend/Frontend :** `FastAPI`, `Streamlit`.
- **Cloud :** `Hugging Face Spaces`, `Streamlit Cloud`.

---

## Équipe du Projet

| <div align="center"><img src="https://github.com/ahmadouniass.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/Khadidiatou1010.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/dior204.png" width="100" style="border-radius:50%"></div> | <div align="center"><img src="https://github.com/Kerencia2.png" width="100" style="border-radius:50%"></div> |
| :---: | :---: | :---: | :---: |
| [**Ahmadou Niass**](https://github.com/ahmadouniass) | [**Khadidiatou Coulibaly**](https://github.com/Khadidiatou1010) | [**Dior Mbengue**](https://github.com/dior204) | [**Pahane S. K. D.**](https://github.com/Kerencia2) |

---
<p align="center">Developed with 💙 by the Team</p>
