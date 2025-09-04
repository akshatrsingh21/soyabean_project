import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

class HighAccuracyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.feature_names = None
        self.accuracy = 0
        self.load_model()
        
        # Known patterns for fallback
        self.variety_patterns = {
            '1110': {'r_range': (30, 135), 'g_range': (29, 119), 'b_range': (25, 77), 'name': '1110 (Early Growth)'},
            '1135': {'r_range': (30, 77), 'g_range': (32, 69), 'b_range': (37, 65), 'name': '1135 (Medium Growth)'}, 
            '2172': {'r_range': (61, 144), 'g_range': (52, 126), 'b_range': (35, 81), 'name': '2172 (Mature)'}
        }
    
    def load_model(self):
        """Load the advanced trained model"""
        try:
            saved_data = joblib.load('soybean_advanced_model.pkl')
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.feature_selector = saved_data['feature_selector']
            self.feature_names = saved_data['feature_names']
            self.accuracy = saved_data.get('accuracy', 0)
            print(f"Advanced model loaded! Accuracy: {self.accuracy:.3f}")
            return True
        except Exception as e:
            print(f"Could not load advanced model: {e}")
            return False
    
    def extract_advanced_features(self, r, g, b):
        """Extract all advanced features used during training"""
        features = {}
        
        # Basic features
        features['Average R'] = r
        features['Average G'] = g
        features['Average B'] = b
        
        # Derived features
        features['Brightness'] = (r + g + b) / 3
        features['R_G_Ratio'] = r / max(g, 1)
        features['R_B_Ratio'] = r / max(b, 1)
        features['G_B_Ratio'] = g / max(b, 1)
        
        # Color dominance
        features['Red_Dominance'] = r / max(g, b, 1)
        features['Green_Dominance'] = g / max(r, b, 1)
        features['Blue_Dominance'] = b / max(r, g, 1)
        
        # Statistical features
        rgb_values = [r, g, b]
        features['RGB_Std'] = np.std(rgb_values)
        features['RGB_Range'] = max(rgb_values) - min(rgb_values)
        features['RGB_Variance'] = np.var(rgb_values)
        
        # Advanced ratios
        total_color = r + g + b
        features['Total_Color'] = total_color
        features['R_Proportion'] = r / total_color
        features['G_Proportion'] = g / total_color
        features['B_Proportion'] = b / total_color
        
        # Differences
        features['RG_Difference'] = r - g
        features['RB_Difference'] = r - b
        features['GB_Difference'] = g - b
        
        # Normalized
        features['R_Normalized'] = r / 255
        features['G_Normalized'] = g / 255
        features['B_Normalized'] = b / 255
        
        # Quadratic and interactions
        features['R_Squared'] = r ** 2
        features['G_Squared'] = g ** 2
        features['B_Squared'] = b ** 2
        features['R_G_Interaction'] = r * g
        features['R_B_Interaction'] = r * b
        features['G_B_Interaction'] = g * b
        
        # Convert to array in correct order
        feature_vector = [features[col] for col in self.feature_names]
        return np.array(feature_vector)
    
    def predict_variety(self, r, g, b):
        """High-accuracy prediction using advanced model"""
        try:
            # Extract all features
            features = self.extract_advanced_features(r, g, b)
            
            # Feature selection
            if self.feature_selector:
                features = self.feature_selector.transform(features.reshape(1, -1))
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            variety_code = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100
            
            variety_name = self.variety_patterns.get(str(variety_code), {}).get('name', str(variety_code))
            
            return f"{variety_name} - {confidence:.1f}% confidence", confidence, variety_code
            
        except Exception as e:
            print(f"Advanced prediction failed: {e}")
            # Fallback to rules
            return self.fallback_prediction(r, g, b)
    
    def fallback_prediction(self, r, g, b):
        """Rule-based fallback"""
        best_match = None
        best_score = 0
        
        for variety, pattern in self.variety_patterns.items():
            score = 0
            
            if pattern['r_range'][0] <= r <= pattern['r_range'][1]:
                score += 1
            if pattern['g_range'][0] <= g <= pattern['g_range'][1]:
                score += 1
            if pattern['b_range'][0] <= b <= pattern['b_range'][1]:
                score += 1
                
            brightness = (r + g + b) / 3
            if variety == '2172' and brightness > 90:
                score += 2
            elif variety == '1110' and 40 <= brightness <= 70:
                score += 2
            elif variety == '1135' and brightness < 50:
                score += 2
                
            if score > best_score:
                best_score = score
                best_match = variety
                
        confidence = min(100, best_score * 25)
        variety_name = self.variety_patterns.get(best_match, {}).get('name', 'Unknown')
        
        return f"{variety_name} - {confidence:.0f}% confidence (Fallback)", confidence, best_match

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.detector = HighAccuracyDetector()

    def run(self):
        try:
            self.progress_signal.emit(10)
            
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.progress_signal.emit(30)
            
            # Calculate average RGB
            r = np.mean(image_rgb[:, :, 0])
            g = np.mean(image_rgb[:, :, 1])
            b = np.mean(image_rgb[:, :, 2])
            
            variety, confidence, variety_code = self.detector.predict_variety(r, g, b)
            self.progress_signal.emit(70)
            
            # Additional analysis
            health_status = self.assess_health(r, g, b, variety_code)
            growth_stage = self.estimate_growth_stage(r, g, b)
            root_analysis = self.analyze_roots(image_rgb)
            self.progress_signal.emit(90)
            
            results = {
                'image_path': self.image_path,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'rgb_values': [round(r, 2), round(g, 2), round(b, 2)],
                'variety': variety,
                'confidence': confidence,
                'brightness': round((r + g + b) / 3, 2),
                'health_status': health_status,
                'growth_stage': growth_stage,
                'root_density': root_analysis['density'],
                'recommendations': self.generate_recommendations(health_status, root_analysis['density'], variety_code)
            }
            
            self.progress_signal.emit(100)
            self.result_signal.emit(results)
            
        except Exception as e:
            self.result_signal.emit({'error': str(e)})
        finally:
            self.finished_signal.emit()

    def analyze_roots(self, image):
        """Enhanced root analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multiple edge detection methods
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        
        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, edges2)
        root_density = (np.sum(combined_edges > 0) / combined_edges.size) * 100
        
        return {'density': round(root_density, 3)}

    def assess_health(self, r, g, b, variety_code):
        """Advanced health assessment"""
        green_ratio = g / max(r, b, 1)
        brightness = (r + g + b) / 3
        
        # Variety-specific health assessment
        if variety_code == '2172':  # Mature variety
            if brightness > 100 and green_ratio > 0.8:
                return "Excellent Health - Prime Condition"
            elif brightness > 80:
                return "Good Health - Maturing Well"
            else:
                return "Needs Attention - Underdeveloped"
                
        elif variety_code == '1110':  # Early growth
            if 40 <= brightness <= 70 and green_ratio > 1.0:
                return "Healthy Growth - On Track"
            elif brightness < 30:
                return "Poor Health - Stunted Growth"
            else:
                return "Normal Development"
                
        else:  # 1135 or unknown
            if green_ratio > 1.1:
                return "Good Health"
            elif green_ratio < 0.9:
                return "Possible Stress"
            else:
                return "Normal Condition"

    def estimate_growth_stage(self, r, g, b):
        """Precise growth stage estimation"""
        brightness = (r + g + b) / 3
        
        if brightness < 35:
            return "Germination Stage (Day 1-2)"
        elif brightness < 50:
            return "Early Growth (Day 3-5)"
        elif brightness < 75:
            return "Vegetative Stage (Day 6-10)"
        elif brightness < 100:
            return "Flowering Stage (Day 11-18)"
        else:
            return "Maturation Stage (Day 19+)"

    def generate_recommendations(self, health_status, root_density, variety_code):
        """Precise recommendations"""
        recommendations = []
        
        if "Excellent" in health_status:
            recommendations.extend(["✓ Optimal conditions maintained", "✓ Continue current regimen"])
        elif "Good" in health_status:
            recommendations.extend(["● Healthy development", "● Monitor growth weekly"])
        else:
            recommendations.extend(["⚠️ Requires attention", "💧 Check moisture levels", "🌿 Assess nutrient balance"])
            
        if root_density < 0.3:
            recommendations.append("🌱 Low root density - improve aeration")
        elif root_density < 0.6:
            recommendations.append("● Moderate root development")
        else:
            recommendations.append("✓ Good root system")
            
        # Variety-specific recommendations
        if variety_code == '2172':
            recommendations.append("🌻 Mature variety - monitor for harvest readiness")
        elif variety_code == '1110':
            recommendations.append("🌱 Early growth - ensure proper nutrients")
            
        return recommendations

class HighAccuracyAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image_path = None

    def initUI(self):
        self.setWindowTitle("High-Accuracy Soybean Analyzer")
        self.setGeometry(100, 100, 1500, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.image_label.setText("No image selected\n\nClick 'Load Image' to begin")
        self.image_label.setFont(QFont("Arial", 12))
        
        # Controls
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("📁 Load Soybean Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        
        self.analyze_btn = QPushButton("🔍 Analyze Image")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.progress_bar)
        control_group.setLayout(control_layout)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(control_group)
        
        # Right panel
        right_panel = QVBoxLayout()
        
        results_group = QGroupBox("📊 High-Accuracy Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: Consolas; font-size: 12px; background-color: #f8f8f8;")
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        right_panel.addWidget(results_group)
        
        layout.addLayout(left_panel, 40)
        layout.addLayout(right_panel, 60)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Soybean Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.setEnabled(True)
            self.results_text.clear()
            self.results_text.append("✅ Image loaded successfully.\n📋 Click 'Analyze Image' for high-accuracy analysis.")

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("❌ Failed to load image")

    def analyze_image(self):
        if not self.current_image_path:
            return
            
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.analysis_thread = AnalysisThread(self.current_image_path)
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.result_signal.connect(self.show_results)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_results(self, results):
        if 'error' in results:
            self.results_text.append(f"❌ Error: {results['error']}")
            return
            
        self.results_text.clear()
        self.results_text.append("🌱 HIGH-ACCURACY SOYBEAN ANALYSIS")
        self.results_text.append("=" * 50)
        self.results_text.append(f"📁 Image: {os.path.basename(results['image_path'])}")
        self.results_text.append(f"📏 Size: {results['image_size']}")
        self.results_text.append("")
        
        self.results_text.append("🎨 PRECISE COLOR ANALYSIS:")
        self.results_text.append(f"   RGB Values: R={results['rgb_values'][0]}, G={results['rgb_values'][1]}, B={results['rgb_values'][2]}")
        self.results_text.append(f"   Brightness: {results['brightness']}")
        self.results_text.append("")
        
        self.results_text.append("🌱 VARIETY IDENTIFICATION:")
        self.results_text.append(f"   {results['variety']}")
        self.results_text.append(f"   Confidence: {results['confidence']:.1f}%")
        self.results_text.append("")
        
        self.results_text.append("📈 GROWTH ANALYSIS:")
        self.results_text.append(f"   Growth Stage: {results['growth_stage']}")
        self.results_text.append(f"   Health Status: {results['health_status']}")
        self.results_text.append("")
        
        self.results_text.append("🌿 ROOT SYSTEM ANALYSIS:")
        self.results_text.append(f"   Root Density: {results['root_density']}%")
        self.results_text.append("   (Professional assessment of root development)")
        self.results_text.append("")
        
        self.results_text.append("💡 EXPERT RECOMMENDATIONS:")
        for rec in results['recommendations']:
            self.results_text.append(f"   • {rec}")

    def analysis_finished(self):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = HighAccuracyAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()