import sys
import os
import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

class EnhancedSoybeanDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.accuracy = 0
        self.load_model()
        
        # Known patterns from your actual data
        self.known_patterns = {
            '1110': {'avg_r': (30, 41), 'avg_g': (29, 42), 'avg_b': (36, 47)},
            '1135': {'avg_r': (30, 46), 'avg_g': (32, 44), 'avg_b': (37, 49)},
            '2172': {'avg_r': (100, 144), 'avg_g': (80, 126), 'avg_b': (45, 81)}
        }
    
    def load_model(self):
        """Load the enhanced model"""
        try:
            saved_data = joblib.load('soybean_enhanced_model.pkl')
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.feature_names = saved_data['feature_names']
            self.accuracy = saved_data.get('accuracy', 0)
            print(f"Enhanced model loaded! Expected accuracy: {self.accuracy:.3f}")
            return True
        except Exception as e:
            print(f"Could not load enhanced model: {e}")
            return False
    
    def extract_features(self, r, g, b):
        """Extract the same features used during training"""
        features = {
            'Average R': r,
            'Average G': g,
            'Average B': b,
            'Brightness': (r + g + b) / 3,
            'R_G_Ratio': r / max(g, 1),
            'R_B_Ratio': r / max(b, 1),
            'G_B_Ratio': g / max(b, 1),
            'Red_Dominance': r / max(g, b, 1),
            'Green_Dominance': g / max(r, b, 1),
            'R_minus_G': r - g,
            'R_minus_B': r - b,
            'G_minus_B': g - b,
            'R_Squared': r ** 2,
            'G_Squared': g ** 2
        }
        
        # Ensure correct order
        return np.array([features[col] for col in self.feature_names])
    
    def predict_variety(self, r, g, b):
        """Predict variety using enhanced model"""
        if self.model is not None:
            try:
                features = self.extract_features(r, g, b)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                
                variety_code = self.label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100
                
                return f"{variety_code} - {confidence:.1f}% confidence", confidence, variety_code
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                return self.fallback_prediction(r, g, b)
        else:
            return self.fallback_prediction(r, g, b)
    
    def fallback_prediction(self, r, g, b):
        """Advanced fallback based on your actual data patterns"""
        # Check which known pattern matches best
        best_match = None
        best_score = 0
        
        for variety, pattern in self.known_patterns.items():
            score = 0
            
            # Check if within known ranges
            if pattern['avg_r'][0] <= r <= pattern['avg_r'][1]:
                score += 1
            if pattern['avg_g'][0] <= g <= pattern['avg_g'][1]:
                score += 1
            if pattern['avg_b'][0] <= b <= pattern['avg_b'][1]:
                score += 1
            
            # Check brightness patterns
            brightness = (r + g + b) / 3
            if variety == '2172' and brightness > 90:
                score += 2
            elif variety == '1110' and 35 <= brightness <= 50:
                score += 2
            elif variety == '1135' and 40 <= brightness <= 55:
                score += 2
                
            if score > best_score:
                best_score = score
                best_match = variety
        
        confidence = min(100, best_score * 25)
        return f"{best_match} - {confidence:.0f}% confidence (Pattern Match)", confidence, best_match

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.detector = EnhancedSoybeanDetector()

    def run(self):
        try:
            self.progress_signal.emit(10)
            
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
            health_status = self.assess_health(r, g, b)
            growth_stage = self.estimate_growth_stage(r, g, b)
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
                'recommendations': self.generate_recommendations(health_status, variety_code)
            }
            
            self.progress_signal.emit(100)
            self.result_signal.emit(results)
            
        except Exception as e:
            self.result_signal.emit({'error': str(e)})
        finally:
            self.finished_signal.emit()

    def assess_health(self, r, g, b):
        """Assess plant health based on color ratios"""
        green_ratio = g / max(r, b, 1)
        brightness = (r + g + b) / 3
        
        if green_ratio > 1.2 and brightness > 50:
            return "Excellent Health"
        elif green_ratio > 1.0 and brightness > 40:
            return "Good Health"
        else:
            return "Needs Attention"

    def estimate_growth_stage(self, r, g, b):
        """Estimate growth stage based on brightness"""
        brightness = (r + g + b) / 3
        
        if brightness < 40:
            return "Early Stage (Day 1-3)"
        elif brightness < 70:
            return "Mid Stage (Day 4-7)"
        elif brightness < 100:
            return "Advanced Stage (Day 8-14)"
        else:
            return "Mature Stage (Day 15+)"

    def generate_recommendations(self, health_status, variety_code):
        """Generate recommendations based on health status and variety"""
        recommendations = []
        
        if "Excellent" in health_status:
            recommendations.extend(["✓ Continue current care routine", "✓ Optimal growing conditions"])
        elif "Good" in health_status:
            recommendations.extend(["● Maintain regular monitoring", "● Good plant health"])
        else:
            recommendations.extend(["⚠️ Check water levels", "⚠️ Assess nutrient needs"])
        
        # Variety-specific recommendations
        if variety_code == '2172':
            recommendations.append("🌻 Mature variety - monitor for harvest readiness")
        elif variety_code == '1110':
            recommendations.append("🌱 Early growth - ensure proper nutrients")
        elif variety_code == '1135':
            recommendations.append("🌿 Medium growth - maintain steady care")
            
        return recommendations

class SoybeanAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image_path = None

    def initUI(self):
        self.setWindowTitle("Enhanced Soybean Variety Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = QVBoxLayout()
        
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
        
        results_group = QGroupBox("Analysis Results")
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
            self.results_text.append("✅ Image loaded successfully.\n📋 Click 'Analyze Image' for analysis.")

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
        self.results_text.append("🌱 ENHANCED SOYBEAN ANALYSIS REPORT")
        self.results_text.append("=" * 50)
        self.results_text.append(f"📁 Image: {os.path.basename(results['image_path'])}")
        self.results_text.append(f"📏 Size: {results['image_size']}")
        self.results_text.append("")
        
        self.results_text.append("🎨 COLOR ANALYSIS:")
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
        
        self.results_text.append("💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            self.results_text.append(f"   • {rec}")

    def analysis_finished(self):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = SoybeanAnalyzerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()