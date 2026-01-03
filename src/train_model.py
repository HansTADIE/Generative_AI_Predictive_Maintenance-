"""
Module d'entraînement du modèle prédictif pour la maintenance
Utilise plusieurs algorithmes et compare leurs performances
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MaintenancePredictor:
    """Classe pour entraîner et évaluer les modèles de prédiction de pannes"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def load_training_data(self, data_dir='../data/train_test'):
        """
        Charge les données d'entraînement et de test
        
        Args:
            data_dir: répertoire contenant les données
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
            y_train = pd.read_csv(os.path.join(data_dir, 'y_train_failure.csv')).values.ravel()
            y_test = pd.read_csv(os.path.join(data_dir, 'y_test_failure.csv')).values.ravel()
            
            print(f"✅ Données chargées :")
            print(f"   - X_train: {X_train.shape}")
            print(f"   - X_test: {X_test.shape}")
            print(f"   - Distribution y_train: {np.bincount(y_train)}")
            print(f"   - Distribution y_test: {np.bincount(y_test)}")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return None, None, None, None
    
    def initialize_models(self):
        """
        Initialise plusieurs modèles pour comparer leurs performances
        """
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Important pour les classes déséquilibrées
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        print(f"\n✅ {len(self.models)} modèles initialisés :")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_model(self, model_name, X_train, y_train):
        """
        Entraîne un modèle spécifique
        
        Args:
            model_name: nom du modèle à entraîner
            X_train: features d'entraînement
            y_train: target d'entraînement
        """
        print(f"\n🔄 Entraînement de {model_name}...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        print(f"✅ {model_name} entraîné")
        
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Évalue les performances d'un modèle
        
        Args:
            model_name: nom du modèle
            model: modèle entraîné
            X_test: features de test
            y_test: target de test
            
        Returns:
            dict avec les métriques de performance
        """
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Stocker les résultats
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Afficher les résultats
        print(f"\n📊 Résultats pour {model_name} :")
        print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall:    {metrics['recall']:.4f}")
        print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   - ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """
        Entraîne et évalue tous les modèles
        
        Args:
            X_train, X_test, y_train, y_test: données d'entraînement et de test
        """
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT ET ÉVALUATION DE TOUS LES MODÈLES")
        print("="*60)
        
        best_f1 = 0
        
        for model_name in self.models.keys():
            # Entraîner le modèle
            model = self.train_model(model_name, X_train, y_train)
            
            # Évaluer le modèle
            metrics = self.evaluate_model(model_name, model, X_test, y_test)
            
            # Garder le meilleur modèle (basé sur F1-score)
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model = model
                self.best_model_name = model_name
        
        print("\n" + "="*60)
        print(f"🏆 MEILLEUR MODÈLE : {self.best_model_name}")
        print(f"   F1-Score : {best_f1:.4f}")
        print("="*60)
    
    def plot_comparison(self, save_path='../models/comparison.png'):
        """
        Crée un graphique comparant les performances des modèles
        
        Args:
            save_path: chemin où sauvegarder le graphique
        """
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        
        # Préparer les données pour le graphique
        data = []
        for model_name in model_names:
            for metric in metrics_names:
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.results[model_name]['metrics'][metric]
                })
        
        df_plot = pd.DataFrame(data)
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model')
        plt.title('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Métrique', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title='Modèle', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graphique de comparaison sauvegardé : {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, model_name=None, save_path='../models/confusion_matrix.png'):
        """
        Affiche la matrice de confusion pour un modèle
        
        Args:
            model_name: nom du modèle (si None, utilise le meilleur)
            save_path: chemin où sauvegarder le graphique
        """
        if model_name is None:
            model_name = self.best_model_name
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pas de panne', 'Panne'],
                    yticklabels=['Pas de panne', 'Panne'])
        plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée : {save_path}")
        plt.close()
    
    def get_feature_importance(self, X_train):
        """
        Affiche l'importance des features pour le meilleur modèle
        
        Args:
            X_train: features d'entraînement (pour les noms de colonnes)
            
        Returns:
            DataFrame avec l'importance des features
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = X_train.columns
            
            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\n📊 Importance des Features ({self.best_model_name}) :")
            print(df_importance.to_string(index=False))
            
            # Graphique
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_importance, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Importance des Features - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            save_path = '../models/feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graphique d'importance sauvegardé : {save_path}")
            plt.close()
            
            return df_importance
        else:
            print(f"⚠️  {self.best_model_name} ne supporte pas feature_importances_")
            return None
    
    def save_best_model(self, save_path='../models/best_model.pkl'):
        """
        Sauvegarde le meilleur modèle
        
        Args:
            save_path: chemin où sauvegarder le modèle
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_info = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]['metrics'],
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_info, save_path)
        print(f"\n✅ Meilleur modèle sauvegardé : {save_path}")
        print(f"   Modèle : {self.best_model_name}")
        print(f"   F1-Score : {model_info['metrics']['f1_score']:.4f}")


def main():
    """Fonction principale pour entraîner les modèles"""
    
    print("🚀 ENTRAÎNEMENT DU MODÈLE PRÉDICTIF DE MAINTENANCE")
    print("="*60)
    
    # Initialiser le predictor
    predictor = MaintenancePredictor()
    
    # Charger les données
    X_train, X_test, y_train, y_test = predictor.load_training_data()
    
    if X_train is None:
        print("❌ Impossible de continuer sans données")
        print("💡 Exécutez d'abord preprocessing.py")
        return
    
    # Initialiser les modèles
    predictor.initialize_models()
    
    # Entraîner et évaluer tous les modèles
    predictor.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Créer les visualisations
    predictor.plot_comparison()
    predictor.plot_confusion_matrix()
    predictor.get_feature_importance(X_train)
    
    # Sauvegarder le meilleur modèle
    predictor.save_best_model()
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print("\n📁 Fichiers générés :")
    print("   - models/best_model.pkl")
    print("   - models/comparison.png")
    print("   - models/confusion_matrix.png")
    print("   - models/feature_importance.png")


if __name__ == "__main__":
    main()