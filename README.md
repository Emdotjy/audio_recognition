# Audio_recognition

Projet d'école réalisé par Simon Vincent et Gasmi Rémy
=======
# Projet de Reconnaissance Vocale : LPC & MFCC avec k-NN et HMM

Ce projet a été réalisé dans le cadre du module **"Traitement et Analyse des Données Visuelles et Sonores"** à l'École Centrale de Lyon. L'objectif était de développer des systèmes simples de reconnaissance de la parole en utilisant deux approches différentes : les **coefficients LPC (Linear Predictive Coding)** avec la méthode des **k plus proches voisins (k-NN)**, et les **coefficients MFCC (Mel-Frequency Cepstral Coefficients)** avec les **Modèles de Markov Cachés (HMM)**.

---

## 📌 Aperçu du Projet

L'objectif principal de ce projet était de reconnaître des chiffres prononcés (de 0 à 9) à partir d'enregistrements audio. Nous avons exploré deux approches distinctes pour résoudre ce problème :

1. **Approche 1 : Coefficients LPC + k-NN**
   - Utilisation des coefficients LPC pour modéliser la parole.
   - Classification avec la méthode des k plus proches voisins (k-NN) et une distance élastique pour gérer les variations de longueur des enregistrements.
   - Résultat : **50% de précision**.

2. **Approche 2 : Coefficients MFCC + HMM**
   - Calcul des coefficients MFCC pour capturer les caractéristiques spectrales des enregistrements.
   - Utilisation de Modèles de Markov Cachés (HMM) pour modéliser les séquences de phonèmes.
   - Optimisation des hyperparamètres
   - Résultat : **85.7% de précision**
---

## 🚀 Fonctionnalités

- **Traitement du signal audio** : Chargement, normalisation et extraction des caractéristiques (LPC et MFCC).
- **Classification** : Utilisation de l'algorithme des k plus proches voisins (k-NN) avec une distance élastique pour la première approche.
- **Modélisation** : Implémentation de Modèles de Markov Cachés (HMM) pour la reconnaissance de séquences audio.
- **Optimisation** : Utilisation de la bibliothèque `numba` pour accélérer les calculs et validation des hyperparamètres pour améliorer les performances.

---

## 📊 Résultats

### Approche 1 : LPC + k-NN
- **Précision** : 50%
- **Matrice de confusion** : Les chiffres 0 et 6 sont les mieux classés, tandis que les chiffres 1 et 5 sont souvent confondus.

### Approche 2 : MFCC + HMM
- **Précision** : 85.7% (avec les hyperparamètres optimisés)
- **Matrice de confusion** : Une amélioration significative a été observée par rapport à la première approche. Bien que les résultats soient déjà solides, une marge de progression existe avec davantage de temps de calcul et de données d'entraînement.

---

## 🛠️ Technologies Utilisées

- **Langage** : Python
- **Bibliothèques** :
  - `librosa` pour le traitement audio.
  - `scipy` et `numpy` pour les calculs scientifiques.
  - `hmmlearn` pour les Modèles de Markov Cachés.
  - `numba` pour l'accélération des calculs.
  - `scikit-learn` pour l'évaluation des performances.

---

## 📂 Structure du Projet

projet-reconnaissance-vocale/
├── data/ # Dossier contenant les enregistrements audio
├── scripts/ # Scripts Python pour le traitement et la classification
│ ├── lpc_knn.py # Script pour l'approche LPC + k-NN
│ ├── mfcc_hmm.py # Script pour l'approche MFCC + HMM
│ └── utils.py # Fonctions utilitaires (chargement des données, découpe du signal en frames,etc.)
├── results/ # Résultats et visualisations (matrices de confusion, etc.)
├── README.md # Ce fichier
└── requirements.txt # Dépendances du projet


---

## 🚀 Comment Exécuter le Projet

1. **Cloner le dépôt** :
    ```bash
    git clone https://github.com/votre-utilisateur/projet-reconnaissance-vocale.git
    cd projet-reconnaissance-vocale

2. **Installer les dépendances** :

    ```bash
    pip install -r requirements.txt

3. **Exécuter les scripts** :

Pour l'approche LPC + k-NN :

    python scripts/lpc_knn.py

Pour l'approche MFCC + HMM :

    python scripts/mfcc_hmm.py


## 📈 Améliorations Futures

- **Base de données plus large** : Utiliser une base de données plus variée avec plus de locuteurs pour améliorer la généralisation du modèle.
- **Modèles de langage** : Intégrer des modèles de langage statistiques ou neuronaux pour améliorer la reconnaissance des mots et des phrases.
- **Optimisation des hyperparamètres** : Explorer davantage les hyperparamètres des HMM (comme `n_components` et `n_iter`) et des MFCC (comme `n_mfcc`, `win_length`, et `hop_length`) pour améliorer les performances.
- **Traitement du bruit** : Ajouter des techniques de réduction du bruit pour améliorer la qualité des enregistrements audio et la robustesse du modèle.
- **Intégration de modèles modernes** : Explorer l'utilisation de réseaux de neurones profonds (RNN, LSTM, ou Transformers) pour capturer des dépendances temporelles plus complexes.
- **Interface utilisateur** : Développer une interface utilisateur simple pour permettre à des non-experts d'utiliser le système de reconnaissance vocale.


---

## 👨‍💻 Auteurs

- **Rémy GASMI** : https://github.com/Mastocodeur
- **Simon VINCENT** : https://github.com/Emdotjy

---


## 🙏 Remerciements

Un grand merci à **Emmanuel DELLANDRÉA** pour son encadrement et ses précieux conseils tout au long de ce projet et bonne lecture à tous !

---

