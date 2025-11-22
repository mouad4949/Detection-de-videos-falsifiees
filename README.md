# 👁️ Détection de Deepfakes par Analyse d'Image (Projet G2)

Ce projet implémente un classifieur d'images basé sur le **Transfert d'Apprentissage** (Deep Learning) capable de distinguer les visages réels des visages générés numériquement (Deepfakes).

L'application finale utilise **Streamlit** pour fournir une interface utilisateur simple et rapide pour tester le modèle entraîné.

---

## ⚙️ Modèle et Performance

| Caractéristique | Description |
|-----------------|-------------|
| **Architecture** | ResNet-50 (Transfert d'Apprentissage) |
| **Méthode d'Entraînement** | Fine-tuning progressif (déblocage de la couche `layer4` et entraînement de la tête de classification personnalisée) |
| **Dataset** | Sous-ensemble de Deepfake and Real Images (Kaggle) |
| **Accuracy Finale (Test)** | 81.22% |
| **Score AUC** | 0.9357 |

---

## 🚀 Comment Exécuter l'Application (Inférence)

Pour lancer l'application Streamlit et tester le modèle entraîné sur vos propres images :

### 1. Structure du Projet

Assurez-vous que les fichiers suivants sont présents dans le même répertoire :

```
.
├── deepfake_detector_notebook.ipynb (Notebook d'entraînement)
├── best_model_finetuned.pth       (Poids du modèle entraîné - 81.22% Acc)
└── app.py                         (Application Streamlit)
```

### 2. Installation des Dépendances

Ce projet nécessite les bibliothèques Python classiques pour le Deep Learning (PyTorch) et Streamlit.

```bash
pip install streamlit torch torchvision timm pillow
```

### 3. Lancement de l'Application

Exécutez le script Streamlit via votre terminal :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement sur `http://localhost:8501`). Vous pouvez y téléverser une image de visage pour obtenir une classification instantanée (`Real` ou `Fake`) ainsi que le score de confiance.

---

## 🧪 Ré-entraînement ou Analyse (Optionnel)

Si vous souhaitez reproduire l'entraînement complet ou analyser la méthodologie en détail :

1. Ouvrez le fichier `deepfake_detector_notebook.ipynb` dans **Google Colab**.
2. Assurez-vous d'utiliser un environnement GPU (`Runtime -> Change Runtime Type`).
3. Suivez les étapes du notebook pour le téléchargement des données, l'entraînement initial, et le Fine-tuning.
