"""
Module de prédiction pour détecter les pannes de machines
Utilisé par l'application Flask pour faire des prédictions en temps réel
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class FailurePredictor:
    """Classe pour effectuer des prédictions de pannes de machines"""
    
    def __init__(self, model_path='../models/best_model.pkl', 
                 scaler_path='../models/scaler.pkl',
                 encoder_path='../models/label_encoder.pkl'):
        """
        Initialise le predictor avec les modèles sauvegardés
        
        Args:
            model_path: chemin vers le modèle entraîné
            scaler_path: chemin vers le scaler
            encoder_path: chemin vers le label encoder
        """
        self.model = None
        self.model_name = None
        self.scaler = None
        self.label_encoder = None
        self.metrics = None
        
        self.load_model(model_path)
        self.load_preprocessors(scaler_path, encoder_path)
    
    def load_model(self, model_path):
        """
        Charge le modèle entraîné
        
        Args:
            model_path: chemin vers le fichier .pkl du modèle
        """
        try:
            model_info = joblib.load(model_path)
            self.model = model_info['model']
            self.model_name = model_info['model_name']
            self.metrics = model_info['metrics']
            
            print(f"✅ Modèle chargé : {self.model_name}")
            print(f"   F1-Score : {self.metrics['f1_score']:.4f}")
            print(f"   Accuracy : {self.metrics['accuracy']:.4f}")
            
        except FileNotFoundError:
            print(f"❌ Modèle non trouvé : {model_path}")
            print("💡 Exécutez d'abord train_model.py")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
    
    def load_preprocessors(self, scaler_path, encoder_path):
        """
        Charge le scaler et l'encoder
        
        Args:
            scaler_path: chemin vers le scaler
            encoder_path: chemin vers l'encoder
        """
        try:
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"✅ Preprocessors chargés")
        except FileNotFoundError as e:
            print(f"❌ Preprocessor non trouvé : {e}")
            print("💡 Exécutez d'abord preprocessing.py")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des preprocessors : {e}")
    
    def prepare_input(self, data):
        """
        Prépare les données d'entrée pour la prédiction
        
        Args:
            data: dict ou DataFrame avec les données de la machine
            
        Returns:
            DataFrame préparé et normalisé
        """
        # Convertir en DataFrame si c'est un dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Encoder le Type si présent
        if 'Type' in df.columns:
            df['Type_encoded'] = self.label_encoder.transform(df['Type'])
        
        # Sélectionner les features dans le bon ordre
        feature_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type_encoded'
        ]
        
        X = df[feature_cols]
        
        # Normaliser
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        return X_scaled
    
    def predict(self, data):
        """
        Fait une prédiction de panne
        
        Args:
            data: dict ou DataFrame avec les données de la machine
            
        Returns:
            dict avec la prédiction et la probabilité
        """
        if self.model is None:
            return {
                'error': 'Modèle non chargé',
                'prediction': None,
                'probability': None
            }
        
        try:
            # Préparer les données
            X = self.prepare_input(data)
            
            # Faire la prédiction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Interpréter les résultats
            result = {
                'prediction': int(prediction),
                'prediction_label': 'PANNE DÉTECTÉE' if prediction == 1 else 'FONCTIONNEMENT NORMAL',
                'probability_failure': float(probability[1]),
                'probability_normal': float(probability[0]),
                'confidence': float(max(probability)),
                'risk_level': self._get_risk_level(probability[1]),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_used': self.model_name
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Erreur lors de la prédiction : {str(e)}',
                'prediction': None,
                'probability': None
            }
    
    def predict_batch(self, data_list):
        """
        Fait des prédictions pour plusieurs machines
        
        Args:
            data_list: liste de dicts avec les données des machines
            
        Returns:
            liste de dicts avec les prédictions
        """
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        
        return results
    
    def _get_risk_level(self, probability):
        """
        Détermine le niveau de risque basé sur la probabilité de panne
        
        Args:
            probability: probabilité de panne (0-1)
            
        Returns:
            str: niveau de risque
        """
        if probability < 0.3:
            return 'FAIBLE'
        elif probability < 0.6:
            return 'MOYEN'
        elif probability < 0.8:
            return 'ÉLEVÉ'
        else:
            return 'CRITIQUE'
    
    def get_diagnostics(self, data):
        """
        Fournit un diagnostic détaillé basé sur les valeurs des capteurs
        
        Args:
            data: dict avec les données de la machine
            
        Returns:
            dict avec les alertes et recommandations
        """
        diagnostics = {
            'alerts': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Vérifier la température de l'air (normal ~300K = 27°C)
        air_temp = data.get('Air temperature [K]', 0)
        if air_temp > 310:  # > 37°C
            diagnostics['alerts'].append(f"Température d'air élevée : {air_temp:.1f}K")
            diagnostics['recommendations'].append("Vérifier le système de refroidissement ambiant")
        
        # Vérifier la température du processus
        process_temp = data.get('Process temperature [K]', 0)
        if process_temp > 320:  # > 47°C
            diagnostics['alerts'].append(f"Température de processus élevée : {process_temp:.1f}K")
            diagnostics['recommendations'].append("Réduire la charge ou améliorer le refroidissement")
        
        # Vérifier la vitesse de rotation
        rpm = data.get('Rotational speed [rpm]', 0)
        if rpm < 1200:
            diagnostics['warnings'].append(f"Vitesse de rotation faible : {rpm:.0f} rpm")
            diagnostics['recommendations'].append("Vérifier le moteur et la transmission")
        elif rpm > 2800:
            diagnostics['warnings'].append(f"Vitesse de rotation élevée : {rpm:.0f} rpm")
            diagnostics['recommendations'].append("Risque de surchauffe, surveiller les vibrations")
        
        # Vérifier le couple
        torque = data.get('Torque [Nm]', 0)
        if torque > 60:
            diagnostics['alerts'].append(f"Couple élevé : {torque:.1f} Nm")
            diagnostics['recommendations'].append("Machine sous forte contrainte, prévoir maintenance préventive")
        
        # Vérifier l'usure de l'outil
        tool_wear = data.get('Tool wear [min]', 0)
        if tool_wear > 200:
            diagnostics['alerts'].append(f"Usure d'outil élevée : {tool_wear:.0f} minutes")
            diagnostics['recommendations'].append("URGENT : Remplacer l'outil de coupe")
        elif tool_wear > 150:
            diagnostics['warnings'].append(f"Usure d'outil modérée : {tool_wear:.0f} minutes")
            diagnostics['recommendations'].append("Planifier le remplacement de l'outil prochainement")
        
        return diagnostics
    
    def generate_summary(self, prediction_result, diagnostics):
        """
        Génère un résumé textuel de l'analyse
        
        Args:
            prediction_result: résultat de la prédiction
            diagnostics: diagnostics détaillés
            
        Returns:
            str: résumé textuel
        """
        summary = []
        
        # En-tête
        summary.append("="*60)
        summary.append("RAPPORT D'ANALYSE PRÉDICTIVE DE MAINTENANCE")
        summary.append("="*60)
        summary.append(f"Date : {prediction_result['timestamp']}")
        summary.append(f"Modèle : {prediction_result['model_used']}")
        summary.append("")
        
        # Prédiction
        summary.append("📊 PRÉDICTION :")
        summary.append(f"   Statut : {prediction_result['prediction_label']}")
        summary.append(f"   Probabilité de panne : {prediction_result['probability_failure']*100:.1f}%")
        summary.append(f"   Niveau de risque : {prediction_result['risk_level']}")
        summary.append(f"   Confiance : {prediction_result['confidence']*100:.1f}%")
        summary.append("")
        
        # Alertes
        if diagnostics['alerts']:
            summary.append("🚨 ALERTES :")
            for alert in diagnostics['alerts']:
                summary.append(f"   - {alert}")
            summary.append("")
        
        # Avertissements
        if diagnostics['warnings']:
            summary.append("⚠️  AVERTISSEMENTS :")
            for warning in diagnostics['warnings']:
                summary.append(f"   - {warning}")
            summary.append("")
        
        # Recommandations
        if diagnostics['recommendations']:
            summary.append("💡 RECOMMANDATIONS :")
            for i, rec in enumerate(diagnostics['recommendations'], 1):
                summary.append(f"   {i}. {rec}")
            summary.append("")
        
        summary.append("="*60)
        
        return "\n".join(summary)


def main():
    """Fonction principale pour tester les prédictions"""
    
    print("🔮 TEST DU MODULE DE PRÉDICTION")
    print("="*60)
    
    # Initialiser le predictor
    predictor = FailurePredictor()
    
    if predictor.model is None:
        print("❌ Impossible de continuer sans modèle chargé")
        return
    
    # Exemple de données pour test
    test_data = {
        'Type': 'M',
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0
    }
    
    print("\n📋 Données de test :")
    for key, value in test_data.items():
        print(f"   {key}: {value}")
    
    # Faire une prédiction
    print("\n🔄 Prédiction en cours...\n")
    result = predictor.predict(test_data)
    
    # Obtenir les diagnostics
    diagnostics = predictor.get_diagnostics(test_data)
    
    # Afficher le résumé
    summary = predictor.generate_summary(result, diagnostics)
    print(summary)
    
    # Test avec une machine à risque
    print("\n" + "="*60)
    print("🧪 TEST AVEC UNE MACHINE À RISQUE")
    print("="*60)
    
    risky_data = {
        'Type': 'H',
        'Air temperature [K]': 312.0,
        'Process temperature [K]': 318.5,
        'Rotational speed [rpm]': 1200,
        'Torque [Nm]': 65.0,
        'Tool wear [min]': 220
    }
    
    print("\n📋 Données de test :")
    for key, value in risky_data.items():
        print(f"   {key}: {value}")
    
    print("\n🔄 Prédiction en cours...\n")
    result2 = predictor.predict(risky_data)
    diagnostics2 = predictor.get_diagnostics(risky_data)
    summary2 = predictor.generate_summary(result2, diagnostics2)
    print(summary2)


if __name__ == "__main__":
    main()