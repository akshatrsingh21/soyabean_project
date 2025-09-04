import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Added missing import

class SimpleEffectiveTrainer:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load the combined dataset"""
        print("Loading combined dataset...")
        self.data = pd.read_excel(self.excel_path)
        print(f"Dataset shape: {self.data.shape}")
        
        # Clean data
        self.data = self.data.dropna(subset=['Variety', 'Average R', 'Average G', 'Average B'])
        print(f"After cleaning: {self.data.shape}")
        
        # Check variety distribution
        print("Variety distribution:")
        print(self.data['Variety'].value_counts())
        
        return self.data
    
    def create_features(self):
        """Create effective features"""
        print("Creating features...")
        
        features = self.data[['Average R', 'Average G', 'Average B']].copy()
        
        # Basic derived features
        features['Brightness'] = (features['Average R'] + features['Average G'] + features['Average B']) / 3
        features['R_G_Ratio'] = features['Average R'] / features['Average G'].replace(0, 1)
        features['R_B_Ratio'] = features['Average R'] / features['Average B'].replace(0, 1)
        features['G_B_Ratio'] = features['Average G'] / features['Average B'].replace(0, 1)
        
        # Color proportions
        total_color = features['Average R'] + features['Average G'] + features['Average B']
        features['R_Proportion'] = features['Average R'] / total_color
        features['G_Proportion'] = features['Average G'] / total_color
        features['B_Proportion'] = features['Average B'] / total_color
        
        # Color differences
        features['RG_Diff'] = features['Average R'] - features['Average G']
        features['RB_Diff'] = features['Average R'] - features['Average B']
        features['GB_Diff'] = features['Average G'] - features['Average B']
        
        print(f"Created {features.shape[1]} features")
        return features
    
    def train_model(self):
        """Train a simple but effective model"""
        # Prepare features and labels
        X = self.create_features()
        y = self.data['Variety'].astype(str)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        # Train Random Forest with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        class_names = [str(cls) for cls in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return accuracy
    
    def save_model(self):
        """Save the trained model"""
        accuracy = self.train_model()
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'accuracy': accuracy
        }, 'soybean_model.pkl')
        print("Model saved as 'soybean_model.pkl'")
        return accuracy
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = self.model.feature_importances_
            feature_names = self.create_features().columns
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            top_features = feature_names[top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(top_importance)), top_features)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Feature importance plot saved as 'feature_importance.png'")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python fixed_trainer.py <path_to_combined_excel_file>")
        return
    
    excel_path = sys.argv[1]
    
    if not os.path.exists(excel_path):
        print(f"File {excel_path} not found!")
        return
    
    # Initialize trainer
    trainer = SimpleEffectiveTrainer(excel_path)
    
    # Load data
    data = trainer.load_data()
    
    # Train and save model
    accuracy = trainer.save_model()
    
    # Plot feature importance
    trainer.plot_feature_importance()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Final Model Accuracy: {accuracy:.3f}")
    
    if accuracy >= 0.8:
        print("🎉 SUCCESS: 80%+ accuracy achieved!")
    else:
        print("⚠️  Accuracy below 80%. Consider adding more data.")

if __name__ == "__main__":
    main()