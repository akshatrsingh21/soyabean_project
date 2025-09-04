import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class FixedExcelTrainer:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_clean_data(self):
        """Load and clean data from Excel"""
        print("Loading Excel data...")
        self.data = pd.read_excel(self.excel_path)
        print(f"Original shape: {self.data.shape}")
        
        # Clean the data
        self.data = self.data.dropna(axis=1, how='all')
        
        # Fill missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # Remove rows with missing variety or RGB values
        self.data = self.data.dropna(subset=['Variety', 'Average R', 'Average G', 'Average B'])
        
        print(f"Cleaned shape: {self.data.shape}")
        print("Variety distribution:")
        print(self.data['Variety'].value_counts())
        
        return self.data
    
    def create_features(self):
        """Create features from RGB values"""
        print("Creating features from RGB data...")
        
        # Basic RGB features
        features = self.data[['Average R', 'Average G', 'Average B']].copy()
        
        # Add derived features
        features['Brightness'] = (features['Average R'] + features['Average G'] + features['Average B']) / 3
        features['R_G_Ratio'] = features['Average R'] / features['Average G'].replace(0, 1)
        features['R_B_Ratio'] = features['Average R'] / features['Average B'].replace(0, 1)
        features['G_B_Ratio'] = features['Average G'] / features['Average B'].replace(0, 1)
        
        # Color dominance features
        features['Red_Dominance'] = features['Average R'] / features[['Average G', 'Average B']].max(axis=1)
        features['Green_Dominance'] = features['Average G'] / features[['Average R', 'Average B']].max(axis=1)
        features['Blue_Dominance'] = features['Average B'] / features[['Average R', 'Average G']].max(axis=1)
        
        # Additional statistical features
        features['RGB_Std'] = features[['Average R', 'Average G', 'Average B']].std(axis=1)
        features['RGB_Range'] = features[['Average R', 'Average G', 'Average B']].max(axis=1) - features[['Average R', 'Average G', 'Average B']].min(axis=1)
        
        print(f"Created {features.shape[1]} features")
        return features
    
    def train_model(self, test_size=0.2):
        """Train the model using RGB features"""
        # Prepare features and labels
        X = self.create_features()
        y = self.data['Variety']
        
        # Convert varieties to strings for proper encoding
        y = y.astype(str)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        
        # FIXED: Convert class names to strings for classification report
        class_names = [str(cls) for cls in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        feature_names = X.columns
        
        print(f"\nTop 10 most important features:")
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        return accuracy
    
    def save_model(self, model_path='soybean_rgb_model.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.create_features().columns.tolist(),
            'accuracy': self.train_model()
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def analyze_variety_patterns(self):
        """Analyze and visualize variety patterns"""
        print("\nAnalyzing variety patterns...")
        
        # Group by variety and calculate mean RGB values
        variety_stats = self.data.groupby('Variety')[['Average R', 'Average G', 'Average B']].mean()
        print("Average RGB values by variety:")
        print(variety_stats)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # RGB values by variety
        plt.subplot(2, 3, 1)
        variety_stats.plot(kind='bar', ax=plt.gca())
        plt.title('Average RGB Values by Variety')
        plt.ylabel('Intensity')
        plt.xticks(rotation=45)
        
        # Scatter plot of R vs G
        plt.subplot(2, 3, 2)
        colors = {'1110': 'red', '1135': 'green', '2172': 'blue'}
        for variety, color in colors.items():
            variety_data = self.data[self.data['Variety'] == int(variety)]
            plt.scatter(variety_data['Average R'], variety_data['Average G'], 
                       alpha=0.6, label=variety, color=color, s=50)
        plt.xlabel('Average R')
        plt.ylabel('Average G')
        plt.title('R vs G by Variety')
        plt.legend()
        plt.grid(True)
        
        # Scatter plot of R vs B
        plt.subplot(2, 3, 3)
        for variety, color in colors.items():
            variety_data = self.data[self.data['Variety'] == int(variety)]
            plt.scatter(variety_data['Average R'], variety_data['Average B'], 
                       alpha=0.6, label=variety, color=color, s=50)
        plt.xlabel('Average R')
        plt.ylabel('Average B')
        plt.title('R vs B by Variety')
        plt.legend()
        plt.grid(True)
        
        # Brightness distribution
        plt.subplot(2, 3, 4)
        self.data['Brightness'] = (self.data['Average R'] + self.data['Average G'] + self.data['Average B']) / 3
        for variety, color in colors.items():
            variety_data = self.data[self.data['Variety'] == int(variety)]
            plt.hist(variety_data['Brightness'], alpha=0.6, label=variety, color=color, bins=15)
        plt.xlabel('Brightness')
        plt.ylabel('Frequency')
        plt.title('Brightness Distribution by Variety')
        plt.legend()
        
        # R/G Ratio
        plt.subplot(2, 3, 5)
        self.data['R_G_Ratio'] = self.data['Average R'] / self.data['Average G']
        for variety, color in colors.items():
            variety_data = self.data[self.data['Variety'] == int(variety)]
            plt.hist(variety_data['R_G_Ratio'], alpha=0.6, label=variety, color=color, bins=15)
        plt.xlabel('R/G Ratio')
        plt.ylabel('Frequency')
        plt.title('R/G Ratio by Variety')
        plt.legend()
        
        # Box plot of RGB values
        plt.subplot(2, 3, 6)
        rgb_data = []
        labels = []
        for variety in [1110, 1135, 2172]:
            variety_data = self.data[self.data['Variety'] == variety]
            rgb_data.append(variety_data['Average R'].values)
            rgb_data.append(variety_data['Average G'].values)
            rgb_data.append(variety_data['Average B'].values)
            labels.extend([f'{variety}-R', f'{variety}-G', f'{variety}-B'])
        
        plt.boxplot(rgb_data, labels=labels)
        plt.xticks(rotation=45)
        plt.title('RGB Distribution by Variety')
        plt.ylabel('Intensity')
        
        plt.tight_layout()
        plt.savefig('variety_patterns_analysis.png', dpi=300, bbox_inches='tight')
        print("Variety patterns analysis saved as 'variety_patterns_analysis.png'")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python fixed_trainer.py <path_to_excel_file>")
        return
    
    excel_path = sys.argv[1]
    
    # Initialize trainer
    trainer = FixedExcelTrainer(excel_path)
    
    # Load and clean data
    data = trainer.load_and_clean_data()
    
    # Analyze variety patterns
    trainer.analyze_variety_patterns()
    
    # Train model
    accuracy = trainer.train_model()
    
    if accuracy is not None:
        # Save model
        joblib.dump({
            'model': trainer.model,
            'scaler': trainer.scaler,
            'label_encoder': trainer.label_encoder,
            'feature_names': trainer.create_features().columns.tolist(),
            'accuracy': accuracy
        }, 'soybean_rgb_model.pkl')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Model Accuracy: {accuracy:.3f}")
        print("Model saved as 'soybean_rgb_model.pkl'")
        print("Variety analysis saved as 'variety_patterns_analysis.png'")

if __name__ == "__main__":
    main()