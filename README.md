# 🤖 IA Générative pour la Maintenance Prédictive

Projet d'intelligence artificielle pour prédire les pannes de machines industrielles et générer automatiquement des rapports de maintenance.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Explication des modules](#explication-des-modules)
- [Résultats attendus](#résultats-attendus)

## 🎯 Vue d'ensemble

Ce projet utilise le **AI4I 2020 Predictive Maintenance Dataset** pour :

1. **Prédire les pannes** de machines industrielles avant qu'elles ne se produisent
2. **Identifier le type de panne** (usure d'outil, surchauffe, défaillance électrique, etc.)
3. **Générer automatiquement** des rapports de maintenance détaillés
4. **Visualiser les résultats** via une interface web Flask

### 🔧 Technologies utilisées

- **Machine Learning** : scikit-learn (Random Forest, Gradient Boosting)
- **Visualisation** : matplotlib, seaborn
- **Web Framework** : Flask
- **IA Générative** : Pour générer des rapports (à venir)

## 📁 Structure du projet

```
Generative_AI_Predictive_Maintenance/
├── data/
│   ├── raw/                        # Données brutes originales
│   │   └── ai4i2020.csv
│   ├── processed/                  # Données augmentées
│   │   └── ai4i_smoted_preserve_props.csv
│   └── train_test/                 # Données préparées pour l'entraînement
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train_failure.csv
│       └── y_test_failure.csv
│
├── models/                         # Modèles entraînés et visualisations
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/                            # Code source
│   ├── preprocessing.py           # Prétraitement des données
│   ├── train_model.py             # Entraînement du modèle
│   └── predict.py                 # Prédictions
│
├── templates/                      # Templates HTML Flask (à venir)
├── static/                         # Fichiers statiques CSS/JS (à venir)
├── app.py                         # Application Flask (à venir)
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone [votre-repo]
cd Generative_AI_Predictive_Maintenance
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Étape 1 : Prétraitement des données

```bash
cd src
python preprocessing.py
```

**Ce que fait ce script :**
- Charge les données augmentées
- Explore et affiche les statistiques
- Encode les variables catégorielles (Type de produit)
- Normalise les features (StandardScaler)
- Divise en ensembles train/test (80/20)
- Sauvegarde tout dans `data/train_test/`

**Résultat attendu :**
```
✅ Données chargées : 10000 lignes, 14 colonnes
✅ Features préparées : 6 colonnes
✅ Données divisées :
   - Entraînement : 8000 échantillons (80%)
   - Test : 2000 échantillons (20%)
✅ Données préparées sauvegardées dans data/train_test/
```

### Étape 2 : Entraînement du modèle

```bash
python train_model.py
```

**Ce que fait ce script :**
- Charge les données préparées
- Entraîne 3 modèles différents :
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
- Compare leurs performances
- Sélectionne automatiquement le meilleur
- Génère des graphiques de visualisation
- Sauvegarde le meilleur modèle

**Résultat attendu :**
```
🏆 MEILLEUR MODÈLE : Random Forest
   F1-Score : 0.9567

📁 Fichiers générés :
   - models/best_model.pkl
   - models/comparison.png
   - models/confusion_matrix.png
   - models/feature_importance.png
```

### Étape 3 : Test de prédiction

```bash
python predict.py
```

**Ce que fait ce script :**
- Charge le modèle entraîné
- Fait des prédictions sur des données de test
- Génère des diagnostics détaillés
- Affiche des recommandations de maintenance

**Résultat attendu :**
```
📊 PRÉDICTION :
   Statut : FONCTIONNEMENT NORMAL
   Probabilité de panne : 5.2%
   Niveau de risque : FAIBLE
   Confiance : 94.8%

💡 RECOMMANDATIONS :
   1. Surveillance normale recommandée
   2. Prochain entretien préventif dans 30 jours
```

## 🔍 Explication des modules

### 📊 preprocessing.py

**Classe principale : `DataPreprocessor`**

Méthodes importantes :
- `load_data()` : Charge le CSV
- `explore_data()` : Affiche les statistiques
- `prepare_features()` : Prépare X et y
- `scale_features()` : Normalise les données
- `split_data()` : Divise train/test

**Features utilisées :**
1. Air temperature [K] - Température ambiante
2. Process temperature [K] - Température du processus
3. Rotational speed [rpm] - Vitesse de rotation
4. Torque [Nm] - Couple/force
5. Tool wear [min] - Usure de l'outil
6. Type_encoded - Type de produit (L/M/H encodé)

### 🤖 train_model.py

**Classe principale : `MaintenancePredictor`**

Méthodes importantes :
- `initialize_models()` : Crée 3 modèles différents
- `train_and_evaluate_all()` : Entraîne et compare
- `plot_comparison()` : Graphique de comparaison
- `get_feature_importance()` : Importance des variables
- `save_best_model()` : Sauvegarde le meilleur

**Métriques évaluées :**
- **Accuracy** : Précision globale
- **Precision** : Qualité des prédictions positives
- **Recall** : Capacité à détecter les pannes
- **F1-Score** : Équilibre precision/recall
- **ROC-AUC** : Performance globale du classifieur

### 🔮 predict.py

**Classe principale : `FailurePredictor`**

Méthodes importantes :
- `predict()` : Fait une prédiction unique
- `predict_batch()` : Prédictions multiples
- `get_diagnostics()` : Analyse détaillée des capteurs
- `generate_summary()` : Rapport textuel complet

**Niveaux de risque :**
- 🟢 **FAIBLE** : < 30% de probabilité
- 🟡 **MOYEN** : 30-60%
- 🟠 **ÉLEVÉ** : 60-80%
- 🔴 **CRITIQUE** : > 80%

## 📈 Résultats attendus

Avec des données bien préparées, vous devriez obtenir :

- **Accuracy** : 95-98%
- **F1-Score** : 90-95%
- **Recall** : 85-95% (important pour détecter les pannes !)
- **Precision** : 90-98%

## 🎯 Prochaines étapes

1. ✅ Modèle prédictif complet (FAIT)
2. ⏳ Interface web Flask
3. ⏳ Génération automatique de rapports avec IA générative
4. ⏳ Prédiction multi-classes (types de pannes)
5. ⏳ API REST pour intégration

## 🐛 Dépannage

### Erreur : "FileNotFoundError"
```bash
# Vérifiez que vous êtes dans le bon répertoire
cd src
# Vérifiez que les données existent
ls ../data/processed/
```

### Erreur : "ModuleNotFoundError"
```bash
# Réinstallez les dépendances
pip install -r requirements.txt
```

### Les performances sont faibles
- Vérifiez la qualité de vos données augmentées
- Augmentez le nombre d'échantillons
- Ajustez les hyperparamètres dans `train_model.py`

## 📚 Ressources

- [Dataset AI4I 2020](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [Guide Flask](https://flask.palletsprojects.com/)

## 👨‍💻 Auteur

Votre nom - Projet IA Maintenance Prédictive

## 📝 Licence

MIT License
