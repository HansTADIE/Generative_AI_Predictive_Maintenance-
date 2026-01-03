"""
Module de prétraitement pour le dataset AI4I 2020 Predictive Maintenance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DataPreprocessor:
    """Classe pour gérer le prétraitement des données de maintenance"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_columns = None
        
    def load_data(self, filepath):
        """
        Charge les données depuis un fichier CSV
        
        Args:
            filepath: chemin vers le fichier CSV
            
        Returns:
            DataFrame pandas avec les données
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return None
    
    def explore_data(self, df):
        """
        Affiche des statistiques descriptives sur les données
        
        Args:
            df: DataFrame pandas
        """
        print("\n" + "="*60)
        print("📊 EXPLORATION DES DONNÉES")
        print("="*60)
        
        print(f"\n🔢 Dimensions : {df.shape}")
        print(f"\n📋 Colonnes : {list(df.columns)}")
        
        print("\n🎯 Distribution des pannes :")
        if 'Machine failure' in df.columns:
            failure_counts = df['Machine failure'].value_counts()
            print(f"   - Sans panne (0) : {failure_counts.get(0, 0)} ({failure_counts.get(0, 0)/len(df)*100:.2f}%)")
            print(f"   - Avec panne (1) : {failure_counts.get(1, 0)} ({failure_counts.get(1, 0)/len(df)*100:.2f}%)")
        
        print("\n🔧 Types de pannes :")
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for ft in failure_types:
            if ft in df.columns:
                count = df[ft].sum()
                print(f"   - {ft} : {count} pannes")
        
        print("\n📈 Statistiques des capteurs :")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe().round(2))
        
        print("\n⚠️  Valeurs manquantes :")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   Aucune valeur manquante ✅")
        else:
            print(missing[missing > 0])
    
    def prepare_features(self, df):
        """
        Prépare les features et les targets pour l'entraînement
        
        Args:
            df: DataFrame pandas
            
        Returns:
            X: Features (DataFrame)
            y_failure: Target principale (panne oui/non)
            y_types: Targets secondaires (types de pannes)
        """
        # Colonnes à utiliser comme features
        feature_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        
        # Encoder le Type de produit (L, M, H)
        if 'Type' in df.columns:
            df['Type_encoded'] = self.label_encoder.fit_transform(df['Type'])
            feature_cols.append('Type_encoded')
        
        # Features numériques
        X = df[feature_cols].copy()
        
        # Target principale : Machine failure
        y_failure = df['Machine failure'].copy() if 'Machine failure' in df.columns else None
        
        # Targets secondaires : types de pannes
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        y_types = df[failure_types].copy() if all(ft in df.columns for ft in failure_types) else None
        
        self.feature_columns = feature_cols
        self.target_columns = ['Machine failure'] + failure_types
        
        print(f"\n✅ Features préparées : {X.shape[1]} colonnes")
        print(f"   Colonnes : {feature_cols}")
        
        return X, y_failure, y_types
    
    def scale_features(self, X_train, X_test):
        """
        Normalise les features avec StandardScaler
        
        Args:
            X_train: Features d'entraînement
            X_test: Features de test
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir en DataFrame pour garder les noms de colonnes
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"✅ Features normalisées (moyenne=0, écart-type=1)")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y_failure, y_types, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test
        
        Args:
            X: Features
            y_failure: Target principale
            y_types: Targets secondaires
            test_size: proportion de données pour le test
            random_state: seed pour la reproductibilité
            
        Returns:
            X_train, X_test, y_train_failure, y_test_failure, y_train_types, y_test_types
        """
        X_train, X_test, y_train_failure, y_test_failure = train_test_split(
            X, y_failure, test_size=test_size, random_state=random_state, stratify=y_failure
        )
        
        if y_types is not None:
            _, _, y_train_types, y_test_types = train_test_split(
                X, y_types, test_size=test_size, random_state=random_state, stratify=y_failure
            )
        else:
            y_train_types, y_test_types = None, None
        
        print(f"\n✅ Données divisées :")
        print(f"   - Entraînement : {X_train.shape[0]} échantillons ({(1-test_size)*100:.0f}%)")
        print(f"   - Test : {X_test.shape[0]} échantillons ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train_failure, y_test_failure, y_train_types, y_test_types
    
    def save_preprocessor(self, save_dir='models'):
        """
        Sauvegarde le scaler et l'encoder pour une utilisation future
        
        Args:
            save_dir: répertoire où sauvegarder les objets
        """
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join('..', 'models', 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join('..', 'models', 'label_encoder.pkl'))
        
        print(f"\n✅ Preprocessor sauvegardé dans {save_dir}/")
    
    def load_preprocessor(self, load_dir='models'):
        """
        Charge le scaler et l'encoder sauvegardés
        
        Args:
            load_dir: répertoire où charger les objets
        """
        self.scaler = joblib.load(os.path.join('..', 'models', 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join('..', 'models', 'label_encoder.pkl'))
        
        print(f"✅ Preprocessor chargé depuis {load_dir}/")


def main():
    """Fonction principale pour tester le preprocessing"""
    
    print("🚀 PREPROCESSING DES DONNÉES DE MAINTENANCE PRÉDICTIVE")
    print("="*60)
    
    # Initialiser le preprocessor
    preprocessor = DataPreprocessor()
    
    # Charger les données
    df = preprocessor.load_data('../data/processed/ai4i_smoted_preserve_props.csv')
    
    if df is None:
        print("❌ Impossible de continuer sans données")
        return
    
    # Explorer les données
    preprocessor.explore_data(df)
    
    # Préparer les features
    X, y_failure, y_types = preprocessor.prepare_features(df)
    
    # Diviser les données
    X_train, X_test, y_train_failure, y_test_failure, y_train_types, y_test_types = \
        preprocessor.split_data(X, y_failure, y_types)
    
    # Normaliser les features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Sauvegarder les données préparées
    os.makedirs('../data/train_test', exist_ok=True)
    
    X_train_scaled.to_csv('../data/train_test/X_train.csv', index=False)
    X_test_scaled.to_csv('../data/train_test/X_test.csv', index=False)
    y_train_failure.to_csv('../data/train_test/y_train_failure.csv', index=False)
    y_test_failure.to_csv('../data/train_test/y_test_failure.csv', index=False)
    
    if y_train_types is not None:
        y_train_types.to_csv('../data/train_test/y_train_types.csv', index=False)
        y_test_types.to_csv('../data/train_test/y_test_types.csv', index=False)
    
    print(f"\n✅ Données préparées sauvegardées dans data/train_test/")
    
    # Sauvegarder le preprocessor
    preprocessor.save_preprocessor()
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING TERMINÉ AVEC SUCCÈS !")
    print("="*60)


if __name__ == "__main__":
    main()