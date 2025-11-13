"""
Enhanced explainability module with better interpretability
Provides meaningful SHAP and LIME explanations with context
"""

import shap
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import json
import os

class EnhancedExplainer:
    """
    Enhanced explainer that provides more meaningful interpretations
    """
    
    def __init__(self, model_path, feature_names, label_map=None, scaler=None):
        """
        Initialize the enhanced explainer
        
        Args:
            model_path: Path to the trained model
            feature_names: List of feature names
            label_map: Dictionary mapping class indices to names
            scaler: Optional scaler for inverse transform
        """
        self.model = joblib.load(model_path)
        self.feature_names = feature_names
        self.label_map = label_map or {}
        self.scaler = scaler
        self.shap_explainer = None
        self.lime_explainer = None
        
    def get_feature_importance_text(self, feature_name, importance_value):
        """
        Convert feature importance to human-readable text
        """
        # Feature descriptions for better understanding
        feature_descriptions = {
            'sbytes': 'Source to destination bytes',
            'dbytes': 'Destination to source bytes',
            'sload': 'Source bits per second',
            'dload': 'Destination bits per second',
            'dur': 'Connection duration',
            'rate': 'Packet rate',
            'sttl': 'Source time to live',
            'dttl': 'Destination time to live',
            'ct_srv_dst': 'Connections to same service/destination',
            'ct_dst_src_ltm': 'Connections from destination to source',
            'service_dns': 'DNS service flag',
            'service_-': 'No service identified',
            'proto_udp': 'UDP protocol flag',
            'synack': 'SYN-ACK flag count',
            'tcprtt': 'TCP round trip time',
            'dpkts': 'Destination packets',
            'dloss': 'Destination packet loss',
            'sinpkt': 'Source inter-packet time',
            'dinpkt': 'Destination inter-packet time',
        }
        
        description = feature_descriptions.get(feature_name, feature_name)
        
        if abs(importance_value) > 0.1:
            strength = "STRONG"
        elif abs(importance_value) > 0.05:
            strength = "MODERATE"
        else:
            strength = "WEAK"
            
        direction = "attack" if importance_value > 0 else "normal"
        
        return f"{description} - {strength} indicator of {direction} traffic"
    
    def explain_shap_detailed(self, features, background_data=None, max_features=10):
        """
        Generate detailed SHAP explanation with context
        
        Args:
            features: Single instance to explain (numpy array or list)
            background_data: Background dataset for SHAP (optional)
            max_features: Number of top features to return
            
        Returns:
            Dictionary with detailed SHAP analysis
        """
        features = np.array(features).reshape(1, -1)
        
        # Initialize SHAP explainer if not cached
        if self.shap_explainer is None:
            try:
                # Try TreeExplainer first (fast)
                self.shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                # Fall back to KernelExplainer
                if background_data is None:
                    background_data = features
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background_data[:100]  # Use subset for speed
                )
        
        # Compute SHAP values
        shap_values = self.shap_explainer.shap_values(features)
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Process SHAP values
        if isinstance(shap_values, list):
            # Multi-class: get values for predicted class
            class_shap_values = shap_values[int(prediction)][0]
        else:
            class_shap_values = shap_values[0]
        
        # Create feature importance list
        importances = []
        for i, (feature_name, shap_val, feature_val) in enumerate(
            zip(self.feature_names, class_shap_values, features[0])
        ):
            importances.append({
                'feature': feature_name,
                'shap_value': float(shap_val),
                'feature_value': float(feature_val),
                'abs_importance': abs(float(shap_val)),
                'interpretation': self.get_feature_importance_text(feature_name, shap_val)
            })
        
        # Sort by absolute importance
        importances.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        # Generate summary
        top_features = importances[:max_features]
        attack_indicators = [f for f in top_features if f['shap_value'] > 0]
        normal_indicators = [f for f in top_features if f['shap_value'] < 0]
        
        summary = {
            'prediction': int(prediction),
            'prediction_label': self.label_map.get(int(prediction), f'Class {prediction}'),
            'confidence': float(max(proba)) if proba is not None else None,
            'top_attack_indicators': [
                {
                    'feature': f['feature'],
                    'value': f['feature_value'],
                    'contribution': f['shap_value'],
                    'interpretation': f['interpretation']
                }
                for f in attack_indicators[:5]
            ],
            'top_normal_indicators': [
                {
                    'feature': f['feature'],
                    'value': f['feature_value'],
                    'contribution': f['shap_value'],
                    'interpretation': f['interpretation']
                }
                for f in normal_indicators[:5]
            ],
            'all_features': top_features,
            'base_value': float(self.shap_explainer.expected_value[int(prediction)]) if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) else float(self.shap_explainer.expected_value)
        }
        
        return summary
    
    def explain_lime_detailed(self, features, background_data, max_features=10):
        """
        Generate detailed LIME explanation
        
        Args:
            features: Single instance to explain
            background_data: Training data for LIME
            max_features: Number of features to explain
            
        Returns:
            Dictionary with detailed LIME analysis
        """
        features = np.array(features).reshape(1, -1)
        
        # Initialize LIME explainer if not cached
        if self.lime_explainer is None:
            self.lime_explainer = LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                class_names=list(self.label_map.values()) if self.label_map else None,
                discretize_continuous=False,  # Don't discretize for better interpretability
                random_state=42
            )
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            features[0],
            self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
            num_features=max_features,
            top_labels=1
        )
        
        # Extract explanation for predicted class
        explanation_list = exp.as_list(label=int(prediction))
        
        # Parse and enhance explanations
        detailed_explanations = []
        for feature_condition, weight in explanation_list:
            # Extract feature name from condition
            feature_name = feature_condition.split()[0]
            
            detailed_explanations.append({
                'feature': feature_name,
                'condition': feature_condition,
                'weight': float(weight),
                'interpretation': self.get_feature_importance_text(feature_name, weight),
                'direction': 'attack' if weight > 0 else 'normal'
            })
        
        summary = {
            'prediction': int(prediction),
            'prediction_label': self.label_map.get(int(prediction), f'Class {prediction}'),
            'confidence': float(max(proba)) if proba is not None else None,
            'explanations': detailed_explanations,
            'intercept': float(exp.intercept[int(prediction)]) if hasattr(exp, 'intercept') else None
        }
        
        return summary
    
    def generate_attack_report(self, features, background_data=None):
        """
        Generate comprehensive attack analysis report
        
        Args:
            features: Instance to analyze
            background_data: Background data for LIME
            
        Returns:
            Dictionary with comprehensive analysis
        """
        features = np.array(features).reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Get SHAP explanation
        shap_analysis = self.explain_shap_detailed(features, background_data)
        
        # Analyze feature values
        feature_stats = []
        for i, (name, value) in enumerate(zip(self.feature_names, features[0])):
            feature_stats.append({
                'feature': name,
                'value': float(value),
                'is_zero': float(value) == 0.0,
                'is_max': float(value) >= 0.99
            })
        
        # Identify key patterns
        zero_features = [f['feature'] for f in feature_stats if f['is_zero']]
        max_features = [f['feature'] for f in feature_stats if f['is_max']]
        
        report = {
            'prediction': {
                'class': int(prediction),
                'label': self.label_map.get(int(prediction), f'Class {prediction}'),
                'confidence': float(max(proba)) if proba is not None else None,
                'is_attack': int(prediction) != 0
            },
            'key_indicators': {
                'attack_signals': shap_analysis['top_attack_indicators'][:3],
                'normal_signals': shap_analysis['top_normal_indicators'][:3]
            },
            'feature_analysis': {
                'total_features': len(self.feature_names),
                'zero_count': len(zero_features),
                'max_count': len(max_features),
                'zero_features': zero_features[:10],
                'suspicious_max_features': max_features[:5]
            },
            'shap_summary': shap_analysis,
            'interpretation': self._generate_interpretation(
                prediction, 
                shap_analysis['top_attack_indicators'], 
                shap_analysis['top_normal_indicators']
            )
        }
        
        return report
    
    def _generate_interpretation(self, prediction, attack_indicators, normal_indicators):
        """
        Generate human-readable interpretation
        """
        threat = self.label_map.get(int(prediction), f'Class {prediction}')
        
        if int(prediction) == 0:
            text = f"This traffic is classified as NORMAL. "
        else:
            text = f"⚠️ This traffic is classified as {threat.upper()} attack. "
        
        if attack_indicators:
            text += f"\n\nKey attack indicators:\n"
            for indicator in attack_indicators[:3]:
                text += f"• {indicator['feature']}: {indicator['interpretation']}\n"
        
        if normal_indicators:
            text += f"\n\nCountering normal indicators:\n"
            for indicator in normal_indicators[:3]:
                text += f"• {indicator['feature']}: {indicator['interpretation']}\n"
        
        return text


if __name__ == "__main__":
    # Example usage
    print("Enhanced Explainability Module")
    print("Import this module in app.py for better explanations")
