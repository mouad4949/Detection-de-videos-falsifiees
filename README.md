Rapport de Projet G2 : Détection de Deepfakes par Réseau Convolutif (ResNet-50)

Auteurs : RGUIBI MOHAMED MOUAD, SABIR ACHRAF, AIT SAID AYOUB

Cycle : S4 - 2024/2025

Résumé

Ce rapport présente la méthodologie et les résultats d'un projet de Vision par Ordinateur visant à développer un classifieur robuste pour la détection d'images falsifiées (deepfakes). Le modèle est basé sur une architecture ResNet-50 exploitant le transfert d'apprentissage et a été optimisé par une stratégie de Fine-tuning progressif. Les résultats finaux démontrent une précision (Accuracy) de plus de $81\%$ et un score AUC (Area Under the Curve) de $0.9357$, validant l'efficacité du modèle pour distinguer le contenu réel du contenu synthétique. L'intégration de ce modèle dans une application Streamlit permet une démonstration en temps réel.

1. Introduction et Objectifs du Projet

Le projet initial (G2) visait à construire une solution d'IA capable d'identifier les \textbf{artefacts subtils de falsification} sur des images ou des vidéos. Pour simplifier la démonstration et le pipeline, l'approche retenue s'est concentrée sur la classification d'images de visages, en utilisant le jeu de données Kaggle \textit{Deepfake and Real Images}.

Le principal défi technique était de développer une méthode d'entraînement permettant au modèle de généraliser son apprentissage et d'atteindre une robustesse élevée, en particulier en distinguant les imperfections du deepfake du bruit de compression normal.

2. Méthodologie et Préparation des Données

2.1. Jeu de Données et Prétraitement

Jeu de Données : \textit{Deepfake and Real Images} (Kaggle).

Taille Totale de Test et Validation : L'ensemble de test utilisé pour l'évaluation finale comprenait 10 905 images.

Transformations : Les images ont été redimensionnées, recadrées (\textit{CenterCrop} $224 \times 224$ pixels) et normalisées en utilisant les moyennes et écarts-types standard d'ImageNet.

Augmentation : Des techniques d'augmentation (\textit{RandomResizedCrop}, \textit{RandomHorizontalFlip}, \textit{RandomRotation}, \textit{ColorJitter}) ont été appliquées pour renforcer la robustesse du modèle.

2.2. Architecture du Modèle

L'approche repose sur le Transfert d'Apprentissage avec le réseau ResNet-50 

.

Backbone : ResNet-50 chargé avec des poids pré-entraînés sur ImageNet.

Tête Personnalisée (\textit{Custom Head}) : Le classifieur final a été remplacé par la séquence suivante pour optimiser la classification binaire :

nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)


3. Phases d'Entraînement

Le processus a utilisé une approche par étapes pour optimiser la convergence et la performance.

3.1. Phase 1 : Entraînement Initial du Classifieur

Stratégie : Seule la tête personnalisée (model.fc) a été entraînée (couches du backbone gelées).

Données : Sous-ensemble rapide de $30\%$ des données d'entraînement.

Résultat : Précision de validation maximale de $\mathbf{84.03\%}$.

3.2. Phase 2 : Fine-Tuning pour la Robustesse

Stratégie : Le dernier bloc de couches profondes de ResNet (model.layer4) a été dégelé.

Données : Augmentation de l'ensemble d'entraînement à $50\%$ pour une meilleure généralisation.

Optimisation : Utilisation de taux d'apprentissage plus faibles pour les couches dégelées ($\mathbf{0.0001}$) afin de stabiliser le Fine-tuning.

Résultat : Amélioration de la précision de validation à $\mathbf{95.48\%}$.

4. Résultats et Évaluation Finale

L'évaluation finale a été réalisée sur l'ensemble de test de $10 905$ images avec le modèle Fine-tuned.

4.1. Rapport de Classification

Classe

Precision

Recall

F1-score

Support

Real

0.74

0.96

0.84

5492

Fake

0.94

0.66

0.78

5413

Accuracy

\multicolumn{4}{

c

}{\textbf{0.81}}



Macro Avg

0.84

0.81

0.81

10905

Analyse des métriques :

Accuracy Finale : $\mathbf{81.22\%}$.

F1-score (Fake) : $0.78$.

La Precision de 0.94 pour la classe Fake est excellente : le modèle est fiable lorsqu'il signale une falsification (peu de Faux Positifs).

Le Recall de 0.66 pour la classe Fake montre le point d'amélioration : le modèle manque encore environ $34\%$ des vrais deepfakes (Faux Négatifs).

4.2. Matrice de Confusion

Vrais Négatifs (TN) : 5275 (Real correctement identifiés).

Faux Positifs (FP) : 217 (Real mal classés en Fake). $\rightarrow$ Très faible.

Faux Négatifs (FN) : 1831 (Fake mal classés en Real). $\rightarrow$ Point d'amélioration.

4.3. Courbe ROC et Courbes d'Apprentissage

AUC Finale : L'AUC est passée de $0.8038$ (Original) à $\mathbf{0.9357}$ (Fine-tuned), confirmant l'efficacité de la stratégie de Fine-tuning.

Courbes d'Apprentissage :  Les courbes montrent une convergence stable sans signes d'overfitting majeur, le Fine-tuning ayant permis une chute significative de la perte.

5. Intégration de l'Application Streamlit

L'application \texttt{app.py} a été développée pour démontrer le modèle en temps réel sur une interface web.

Chargement : Le modèle ResNet-50 Fine-tuned (best_model_finetuned.pth) est chargé et restauré avec sa tête personnalisée exacte.

Fonctionnalité : L'utilisateur téléverse une image, et le modèle effectue le pré-traitement, l'inférence et affiche la classification (Real ou Fake) ainsi que les scores de probabilité.

6. Conclusion

Le projet a abouti à un classifieur de deepfakes robuste, atteignant une précision de $81.22\%$ et un score AUC de $0.9357$. Le modèle est prêt pour une intégration dans un environnement de test ou une utilisation en production pour la détection d'images.
