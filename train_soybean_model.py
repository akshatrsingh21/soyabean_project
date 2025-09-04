import pandas as pd
import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skimage import feature, exposure
import warnings
warnings.filterwarnings('ignore')

class SoybeanModelTrainer:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and clean the Excel data"""
        print("Loading Excel data...")
        self.data = pd.read_excel(self.excel_path)
        print(f"Original data shape: {self.data.shape}")
        
        # Clean the data
        self.clean_data()
        return self.data
    
    def clean_data(self):
        """Clean the dataset"""
        # Remove completely empty columns
        self.data = self.data.dropna(axis=1, how='all')
        
        # Fill missing numeric values with 0
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # Remove rows with missing essential data
        self.data = self.data.dropna(subset=['Image', 'Variety'])
        
        print(f"Cleaned data shape: {self.data.shape}")
        print(f"Variety distribution:\n{self.data['Variety'].value_counts()}")
    
    def extract_features_from_image(self, image_path):
        """Extract features from a single image"""
        try:
            if not os.path.exists(image_path):
                # Try to find the image in different locations
                basename = os.path.basename(image_path)
                possible_paths = [
                    image_path,
                    os.path.join(os.path.dirname(self.excel_path), basename),
                    os.path.join('.', basename),
                    os.path.join('images', basename)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break
                else:
                    print(f"Image not found: {basename}")
                    return None
            
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract comprehensive features
            features = []
            
            # 1. Basic color features (most important)
            for i in range(3):  # R, G, B channels
                channel = image_rgb[:, :, i]
                features.append(np.mean(channel))   # Mean
                features.append(np.std(channel))    # Standard deviation
                features.append(np.median(channel)) # Median
                features.append(np.max(channel))    # Maximum
                features.append(np.min(channel))    # Minimum
            
            # 2. Color ratios (very important for variety detection)
            r_mean = np.mean(image_rgb[:, :, 0])
            g_mean = np.mean(image_rgb[:, :, 1])
            b_mean = np.mean(image_rgb[:, :, 2])
            
            features.append(r_mean / max(g_mean, 1))  # R/G ratio
            features.append(r_mean / max(b_mean, 1))  # R/B ratio
            features.append(g_mean / max(b_mean, 1))  # G/B ratio
            features.append((r_mean + g_mean + b_mean) / 3)  # Overall brightness
            
            # 3. HSV color space features
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for i in range(3):
                features.append(np.mean(hsv[:, :, i]))
            
            # 4. Texture features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # GLCM texture
            try:
                glcm = feature.graycomatrix(gray, [1], [0], symmetric=True, normed=True)
                for prop in ['contrast', 'energy', 'homogeneity']:
                    features.append(feature.graycoprops(glcm, prop)[0, 0])
            except:
                features.extend([0, 0, 0])
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features.append(np.sum(edges > 0) / edges.size)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def prepare_training_data(self):
        """Prepare features and labels for training"""
        print("Preparing training data...")
        
        features_list = []
        labels_list = []
        failed_images = 0
        
        for idx, row in self.data.iterrows():
            image_path = row['Image']
            variety = row['Variety']
            
            # Extract features from image
            features = self.extract_features_from_image(str(image_path))
            
            if features is not None:
                features_list.append(features)
                labels_list.append(variety)
            else:
                failed_images += 1
        
        print(f"Successfully processed {len(features_list)} images")
        print(f"Failed to process {failed_images} images")
        
        return np.array(features_list), np.array(labels_list)
    
    def train_model(self, test_size=0.2):
        """Train the machine learning model"""
        # Prepare data
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No valid training data found!")
            return None
        
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
            class_weight='balanced'  # Handle class imbalance
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        print(f"\nTop 10 most important features:")
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        return accuracy
    
    def save_model(self, model_path='soybean_model.pkl'):
        """Save the trained model and preprocessing objects"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='soybean_model.pkl'):
        """Load a trained model"""
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            print("Model loaded successfully!")
            return True
        except:
            print("No trained model found.")
            return False

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python train_soybean_model.py <path_to_excel_file>")
        return
    
    excel_path = sys.argv[1]
    
    # Initialize trainer
    trainer = SoybeanModelTrainer(excel_path)
    
    # Load data
    data = trainer.load_data()
    
    # Train model
    accuracy = trainer.train_model()
    
    if accuracy is not None:
        # Save model
        trainer.save_model()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final Model Accuracy: {accuracy:.3f}")
        print("Model saved as 'soybean_model.pkl'")
        print("You can now use the analysis tool with the trained model.")

if __name__ == "__main__":
    main()