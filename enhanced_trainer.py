import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EnhancedSoybeanTrainer:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_analyze_data(self):
        """Load and analyze your actual data"""
        print("Loading your soybean data...")
        self.data = pd.read_excel(self.excel_path)
        print(f"Original shape: {self.data.shape}")
        
        # Clean and prepare data
        self.data = self.data[['Variety', 'Average R', 'Average G', 'Average B']].dropna()
        self.data['Variety'] = self.data['Variety'].astype(str)
        
        print(f"Final shape: {self.data.shape}")
        print("Variety distribution:")
        print(self.data['Variety'].value_counts())
        
        # Analyze actual data patterns
        self.analyze_data_patterns()
        
        return self.data
    
    def analyze_data_patterns(self):
        """Analyze the actual patterns in your data"""
        print("\nAnalyzing data patterns...")
        
        # Group by variety and show statistics
        variety_stats = self.data.groupby('Variety').agg({
            'Average R': ['mean', 'std', 'min', 'max'],
            'Average G': ['mean', 'std', 'min', 'max'], 
            'Average B': ['mean', 'std', 'min', 'max']
        })
        
        print("Variety statistics:")
        print(variety_stats)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # RGB values by variety
        plt.subplot(2, 2, 1)
        for variety in self.data['Variety'].unique():
            variety_data = self.data[self.data['Variety'] == variety]
            plt.scatter(variety_data['Average R'], variety_data['Average G'], 
                       alpha=0.6, label=variety, s=50)
        plt.xlabel('Average R')
        plt.ylabel('Average G')
        plt.title('R vs G by Variety (Your Data)')
        plt.legend()
        plt.grid(True)
        
        # Distribution of R values
        plt.subplot(2, 2, 2)
        for variety in self.data['Variety'].unique():
            variety_data = self.data[self.data['Variety'] == variety]
            plt.hist(variety_data['Average R'], alpha=0.6, label=variety, bins=15)
        plt.xlabel('Average R')
        plt.ylabel('Frequency')
        plt.title('R Value Distribution')
        plt.legend()
        
        # Brightness distribution
        plt.subplot(2, 2, 3)
        self.data['Brightness'] = (self.data['Average R'] + self.data['Average G'] + self.data['Average B']) / 3
        for variety in self.data['Variety'].unique():
            variety_data = self.data[self.data['Variety'] == variety]
            plt.hist(variety_data['Brightness'], alpha=0.6, label=variety, bins=15)
        plt.xlabel('Brightness')
        plt.ylabel('Frequency')
        plt.title('Brightness Distribution')
        plt.legend()
        
        # R/G Ratio
        plt.subplot(2, 2, 4)
        self.data['R_G_Ratio'] = self.data['Average R'] / self.data['Average G']
        for variety in self.data['Variety'].unique():
            variety_data = self.data[self.data['Variety'] == variety]
            plt.hist(variety_data['R_G_Ratio'], alpha=0.6, label=variety, bins=15)
        plt.xlabel('R/G Ratio')
        plt.ylabel('Frequency')
        plt.title('R/G Ratio Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('your_data_analysis.png', dpi=300, bbox_inches='tight')
        print("Data analysis saved as 'your_data_analysis.png'")
    
    def create_advanced_features(self):
        """Create features based on your actual data patterns"""
        print("Creating advanced features...")
        
        features = self.data[['Average R', 'Average G', 'Average B']].copy()
        
        # Based on your data analysis, create features that should separate the varieties
        features['Brightness'] = (features['Average R'] + features['Average G'] + features['Average B']) / 3
        features['R_G_Ratio'] = features['Average R'] / features['Average G'].replace(0, 1)
        features['R_B_Ratio'] = features['Average R'] / features['Average B'].replace(0, 1)
        features['G_B_Ratio'] = features['Average G'] / features['Average B'].replace(0, 1)
        
        # Color dominance features
        features['Red_Dominance'] = features['Average R'] / features[['Average G', 'Average B']].max(axis=1)
        features['Green_Dominance'] = features['Average G'] / features[['Average R', 'Average B']].max(axis=1)
        
        # Special features based on your data patterns
        features['R_minus_G'] = features['Average R'] - features['Average G']
        features['R_minus_B'] = features['Average R'] - features['Average B']
        features['G_minus_B'] = features['Average G'] - features['Average B']
        
        # Quadratic features
        features['R_Squared'] = features['Average R'] ** 2
        features['G_Squared'] = features['Average G'] ** 2
        
        print(f"Created {features.shape[1]} features")
        return features
    
    def train_model_with_cross_validation(self):
        """Train model with extensive cross-validation"""
        # Prepare features and labels
        X = self.create_advanced_features()
        y = self.data['Variety']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle class imbalance with SMOTE
        print("Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_resampled)
        
        # Extensive cross-validation
        print("Performing extensive cross-validation...")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold CV
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            X_scaled, y_resampled, cv=cv, scoring='accuracy'
        )
        
        print(f"Cross-validation scores: {[f'{score:.3f}' for score in cv_scores]}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on all data
        print("Training final model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_scaled, y_resampled)
        
        return cv_scores.mean()
    
    def save_model(self):
        """Save the trained model"""
        accuracy = self.train_model_with_cross_validation()
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'accuracy': accuracy,
            'feature_names': self.create_advanced_features().columns.tolist()
        }, 'soybean_enhanced_model.pkl')
        print("Enhanced model saved as 'soybean_enhanced_model.pkl'")
        return accuracy

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python enhanced_trainer.py <path_to_your_excel_file>")
        return
    
    excel_path = sys.argv[1]
    
    if not os.path.exists(excel_path):
        print(f"File {excel_path} not found!")
        return
    
    # Initialize trainer
    trainer = EnhancedSoybeanTrainer(excel_path)
    
    # Load and analyze data
    data = trainer.load_and_analyze_data()
    
    # Train and save model
    accuracy = trainer.save_model()
    
    print("\n" + "="*60)
    print("ENHANCED TRAINING COMPLETED!")
    print("="*60)
    print(f"Final Model Accuracy: {accuracy:.3f}")
    
    if accuracy >= 0.8:
        print("🎉 SUCCESS: 80%+ accuracy achieved!")
    else:
        print("⚠️  Accuracy below 80%. The varieties may be too similar in RGB space.")
        print("💡 Consider: ")
        print("   - Collecting more diverse images")
        print("   - Using additional features beyond RGB")
        print("   - Taking images under consistent lighting conditions")

if __name__ == "__main__":
    main()