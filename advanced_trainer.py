import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedSoybeanTrainer:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        
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
    
    def create_advanced_features(self):
        """Create advanced features from RGB values"""
        print("Creating advanced features from RGB data...")
        
        # Basic RGB features
        features = self.data[['Average R', 'Average G', 'Average B']].copy()
        
        # 1. Basic derived features
        features['Brightness'] = (features['Average R'] + features['Average G'] + features['Average B']) / 3
        features['R_G_Ratio'] = features['Average R'] / features['Average G'].replace(0, 1)
        features['R_B_Ratio'] = features['Average R'] / features['Average B'].replace(0, 1)
        features['G_B_Ratio'] = features['Average G'] / features['Average B'].replace(0, 1)
        
        # 2. Color dominance features
        features['Red_Dominance'] = features['Average R'] / features[['Average G', 'Average B']].max(axis=1)
        features['Green_Dominance'] = features['Average G'] / features[['Average R', 'Average B']].max(axis=1)
        features['Blue_Dominance'] = features['Average B'] / features[['Average R', 'Average G']].max(axis=1)
        
        # 3. Statistical features
        features['RGB_Std'] = features[['Average R', 'Average G', 'Average B']].std(axis=1)
        features['RGB_Range'] = features[['Average R', 'Average G', 'Average B']].max(axis=1) - features[['Average R', 'Average G', 'Average B']].min(axis=1)
        features['RGB_Variance'] = features[['Average R', 'Average G', 'Average B']].var(axis=1)
        
        # 4. Advanced ratio features
        features['Total_Color'] = features['Average R'] + features['Average G'] + features['Average B']
        features['R_Proportion'] = features['Average R'] / features['Total_Color']
        features['G_Proportion'] = features['Average G'] / features['Total_Color']
        features['B_Proportion'] = features['Average B'] / features['Total_Color']
        
        # 5. Color difference features
        features['RG_Difference'] = features['Average R'] - features['Average G']
        features['RB_Difference'] = features['Average R'] - features['Average B']
        features['GB_Difference'] = features['Average G'] - features['Average B']
        
        # 6. Normalized features
        features['R_Normalized'] = features['Average R'] / 255
        features['G_Normalized'] = features['Average G'] / 255
        features['B_Normalized'] = features['Average B'] / 255
        
        # 7. Quadratic features (interaction terms)
        features['R_Squared'] = features['Average R'] ** 2
        features['G_Squared'] = features['Average G'] ** 2
        features['B_Squared'] = features['Average B'] ** 2
        features['R_G_Interaction'] = features['Average R'] * features['Average G']
        features['R_B_Interaction'] = features['Average R'] * features['Average B']
        features['G_B_Interaction'] = features['Average G'] * features['Average B']
        
        print(f"Created {features.shape[1]} advanced features")
        return features
    
    def feature_selection(self, X, y):
        """Select best features using ANOVA F-value"""
        print("Performing feature selection...")
        selector = SelectKBest(score_func=f_classif, k=15)
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Get selected feature names
        feature_names = X.columns
        selected_mask = selector.get_support()
        selected_features = feature_names[selected_mask]
        
        print("Selected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
        
        return X_selected, selected_features
    
    def train_advanced_model(self, test_size=0.2):
        """Train advanced model with hyperparameter tuning"""
        # Prepare features and labels
        X = self.create_advanced_features()
        y = self.data['Variety'].astype(str)  # Convert to string for proper encoding
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Feature selection
        X_selected, selected_features = self.feature_selection(X, y_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        # Test multiple advanced models
        models = {
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'LightGBM': LGBMClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        best_accuracy = 0
        best_model = None
        best_model_name = ""
        
        print("\nTesting multiple advanced models...")
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} - CV Accuracy: {cv_mean:.3f}, Test Accuracy: {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        # Hyperparameter tuning for best model
        print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
        
        if best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif best_model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63]
            }
        else:  # RandomForest as default
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        
        # Final evaluation
        y_pred = best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Model Accuracy: {final_accuracy:.3f}")
        print("\nClassification Report:")
        class_names = [str(cls) for cls in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            print(f"\nTop 10 most important features:")
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"{i+1}. {selected_features[idx]}: {feature_importance[idx]:.4f}")
        
        self.best_model = best_model
        return final_accuracy
    
    def save_model(self, model_path='soybean_advanced_model.pkl'):
        """Save the trained model with all components"""
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'accuracy': self.train_advanced_model(),
            'feature_names': self.create_advanced_features().columns.tolist()
        }, model_path)
        print(f"Advanced model saved to {model_path}")
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis visualizations"""
        print("\nCreating comprehensive analysis...")
        
        # Load data if not already loaded
        if self.data is None:
            self.load_and_clean_data()
        
        # Create advanced features
        X = self.create_advanced_features()
        y = self.data['Variety']
        
        plt.figure(figsize=(20, 15))
        
        # 1. Correlation heatmap
        plt.subplot(2, 3, 1)
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        # 2. Pairplot of top features
        plt.subplot(2, 3, 2)
        top_features = X.columns[:4]  # First 4 features
        for i, feature in enumerate(top_features):
            plt.subplot(2, 2, i+1)
            for variety in y.unique():
                variety_data = X[y == variety]
                plt.hist(variety_data[feature], alpha=0.6, label=str(variety), bins=15)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
        plt.suptitle('Distribution of Top Features by Variety')
        
        # 3. 3D scatter plot (if possible)
        try:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.subplot(2, 3, 3, projection='3d')
            colors = {'1110': 'red', '1135': 'green', '2172': 'blue'}
            for variety, color in colors.items():
                variety_data = X[y == int(variety)]
                ax.scatter(variety_data['Average R'], variety_data['Average G'], variety_data['Average B'], 
                          alpha=0.6, label=variety, color=color, s=30)
            ax.set_xlabel('Average R')
            ax.set_ylabel('Average G')
            ax.set_zlabel('Average B')
            ax.set_title('3D RGB Space - Variety Separation')
            ax.legend()
        except:
            plt.subplot(2, 3, 3)
            plt.text(0.5, 0.5, '3D plot not available', ha='center', va='center')
            plt.title('3D Visualization')
        
        # 4. Feature importance visualization
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 3, 4)
            feature_importance = self.best_model.feature_importances_
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            top_features = X.columns[top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(top_importance)), top_features)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("Comprehensive analysis saved as 'comprehensive_analysis.png'")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_trainer.py <path_to_excel_file>")
        return
    
    excel_path = sys.argv[1]
    
    # Initialize advanced trainer
    trainer = AdvancedSoybeanTrainer(excel_path)
    
    # Load and clean data
    data = trainer.load_and_clean_data()
    
    # Train advanced model
    accuracy = trainer.train_advanced_model()
    
    if accuracy is not None:
        # Save model
        trainer.save_model()
        
        # Create comprehensive analysis
        trainer.create_comprehensive_analysis()
        
        print("\n" + "="*70)
        print("ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Final Model Accuracy: {accuracy:.3f}")
        print("Advanced model saved as 'soybean_advanced_model.pkl'")
        print("Comprehensive analysis saved as 'comprehensive_analysis.png'")
        
        if accuracy >= 0.8:
            print("🎉 Target achieved: 80%+ accuracy!")
        else:
            print("⚠️  Target not achieved. Consider collecting more data or feature engineering.")

if __name__ == "__main__":
    main()