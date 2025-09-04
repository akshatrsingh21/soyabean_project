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

class ImprovedVarietyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.load_model()
        
        # Known patterns from your data analysis
        self.variety_patterns = {
            '1110': {'r_range': (30, 135), 'g_range': (29, 119), 'b_range': (25, 77), 'name': '1110 (Early Growth)'},
            '1135': {'r_range': (30, 77), 'g_range': (32, 69), 'b_range': (37, 65), 'name': '1135 (Medium Growth)'}, 
            '2172': {'r_range': (61, 144), 'g_range': (52, 126), 'b_range': (35, 81), 'name': '2172 (Mature)'}
        }
    
    def load_model(self):
        """Load the trained RGB model"""
        try:
            saved_data = joblib.load('soybean_rgb_model.pkl')
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.feature_names = saved_data['feature_names']
            self.accuracy = saved_data.get('accuracy', 0.65)
            print(f"Model loaded! Training accuracy: {self.accuracy:.3f}")
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
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
            'Blue_Dominance': b / max(r, g, 1),
            'RGB_Std': np.std([r, g, b]),
            'RGB_Range': max(r, g, b) - min(r, g, b)
        }
        
        return np.array([features[col] for col in self.feature_names])
    
    def predict_with_model(self, r, g, b):
        """Predict using the trained model"""
        try:
            features = self.extract_features(r, g, b).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            variety_code = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100
            
            return variety_code, confidence, probabilities
            
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return None, 0, None
    
    def predict_with_rules(self, r, g, b):
        """Rule-based prediction based on your data patterns"""
        # Check which variety pattern matches best
        best_match = None
        best_score = 0
        
        for variety, pattern in self.variety_patterns.items():
            score = 0
            
            # Check if within known ranges
            if pattern['r_range'][0] <= r <= pattern['r_range'][1]:
                score += 1
            if pattern['g_range'][0] <= g <= pattern['g_range'][1]:
                score += 1
            if pattern['b_range'][0] <= b <= pattern['b_range'][1]:
                score += 1
                
            # Check brightness patterns (from your data analysis)
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
                
        if best_match and best_score >= 2:
            confidence = min(100, best_score * 25)  # Convert score to percentage
            return best_match, confidence, None
            
        return "Unknown", 50, None
    
    def predict_variety(self, r, g, b):
        """Main prediction method"""
        # Try model first
        if self.model is not None:
            variety, confidence, probs = self.predict_with_model(r, g, b)
            if variety is not None and confidence > 50:
                variety_name = self.variety_patterns.get(str(variety), {}).get('name', str(variety))
                return f"{variety_name} - {confidence:.1f}% confidence (Model)"
        
        # Fallback to rules
        variety, confidence, _ = self.predict_with_rules(r, g, b)
        variety_name = self.variety_patterns.get(variety, {}).get('name', variety)
        return f"{variety_name} - {confidence:.0f}% confidence (Rules)"

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.detector = ImprovedVarietyDetector()

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
            
            variety = self.detector.predict_variety(r, g, b)
            self.progress_signal.emit(70)
            
            # Additional analysis
            health_status = self.assess_health(r, g, b)
            growth_stage = self.estimate_growth_stage(r, g, b)
            root_analysis = self.analyze_roots(image_rgb)
            self.progress_signal.emit(90)
            
            results = {
                'image_path': self.image_path,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'rgb_values': [round(r, 2), round(g, 2), round(b, 2)],
                'variety': variety,
                'brightness': round((r + g + b) / 3, 2),
                'health_status': health_status,
                'growth_stage': growth_stage,
                'root_density': root_analysis['density'],
                'recommendations': self.generate_recommendations(health_status, root_analysis['density'])
            }
            
            self.progress_signal.emit(100)
            self.result_signal.emit(results)
            
        except Exception as e:
            self.result_signal.emit({'error': str(e)})
        finally:
            self.finished_signal.emit()

    def analyze_roots(self, image):
        """Root analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        root_density = (np.sum(edges > 0) / edges.size) * 100
        
        return {'density': round(root_density, 3)}

    def assess_health(self, r, g, b):
        """Health assessment based on color ratios"""
        green_ratio = g / max(r, b, 1)
        brightness = (r + g + b) / 3
        
        if green_ratio > 1.2 and brightness > 50:
            return "Excellent Health"
        elif green_ratio > 1.0 and brightness > 40:
            return "Good Health"
        elif brightness < 30:
            return "Poor Health - Low Light"
        else:
            return "Needs Monitoring"

    def estimate_growth_stage(self, r, g, b):
        """Growth stage estimation"""
        brightness = (r + g + b) / 3
        
        if brightness < 40:
            return "Early Stage (Day 1-3)"
        elif brightness < 70:
            return "Mid Stage (Day 4-7)"
        elif brightness < 100:
            return "Advanced Stage (Day 8-14)"
        else:
            return "Mature Stage (Day 15+)"

    def generate_recommendations(self, health_status, root_density):
        """Generate recommendations"""
        recommendations = []
        
        if "Excellent" in health_status:
            recommendations.extend(["✓ Optimal growing conditions", "✓ Continue current care routine"])
        elif "Good" in health_status:
            recommendations.extend(["● Good plant health", "● Maintain regular monitoring"])
        else:
            recommendations.extend(["⚠️ Requires attention", "💧 Check water levels", "🌿 Assess nutrient needs"])
            
        if root_density < 0.5:
            recommendations.append("🌱 Consider root-stimulating treatments")
            
        return recommendations

class ResultsVisualization(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def create_visualization(self, rgb_values, variety):
        """Create a visualization of the RGB analysis"""
        self.fig.clear()
        
        r, g, b = rgb_values
        colors = ['red', 'green', 'blue']
        values = [r, g, b]
        
        # Create bar chart
        ax = self.fig.add_subplot(111)
        bars = ax.bar(['Red', 'Green', 'Blue'], values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom')
        
        ax.set_ylabel('Intensity')
        ax.set_title(f'RGB Analysis - {variety.split("-")[0]}')
        ax.grid(True, alpha=0.3)
        
        self.draw()

class ImprovedSoybeanAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image_path = None

    def initUI(self):
        self.setWindowTitle("Improved Soybean Variety Analyzer")
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        # Visualization
        self.visualization = ResultsVisualization(self, width=4, height=3, dpi=100)
        self.visualization.setMinimumSize(400, 300)
        
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
        left_panel.addWidget(self.visualization)
        left_panel.addWidget(control_group)
        
        # Right panel
        right_panel = QVBoxLayout()
        
        results_group = QGroupBox("📊 Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: Consolas; font-size: 12px; background-color: #f8f8f8;")
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        right_panel.addWidget(results_group)
        
        layout.addLayout(left_panel, 45)
        layout.addLayout(right_panel, 55)

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
            self.results_text.append("✅ Image loaded successfully.\n📋 Click 'Analyze Image' for detailed analysis.")

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
        self.results_text.append("🌱 IMPROVED SOYBEAN ANALYSIS REPORT")
        self.results_text.append("=" * 50)
        self.results_text.append(f"📁 Image: {os.path.basename(results['image_path'])}")
        self.results_text.append(f"📏 Size: {results['image_size']}")
        self.results_text.append("")
        
        self.results_text.append("🎨 COLOR ANALYSIS:")
        self.results_text.append(f"   RGB Values: R={results['rgb_values'][0]}, G={results['rgb_values'][1]}, B={results['rgb_values'][2]}")
        self.results_text.append(f"   Brightness: {results['brightness']}")
        self.results_text.append("")
        
        self.results_text.append("🌱 PLANT ANALYSIS:")
        self.results_text.append(f"   Variety: {results['variety']}")
        self.results_text.append(f"   Growth Stage: {results['growth_stage']}")
        self.results_text.append(f"   Health Status: {results['health_status']}")
        self.results_text.append("")
        
        self.results_text.append("🌿 ROOT ANALYSIS:")
        self.results_text.append(f"   Root Density: {results['root_density']}%")
        self.results_text.append("   (Higher percentage indicates better root development)")
        self.results_text.append("")
        
        self.results_text.append("💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            self.results_text.append(f"   {rec}")
        
        # Update visualization
        self.visualization.create_visualization(results['rgb_values'], results['variety'])

    def analysis_finished(self):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = ImprovedSoybeanAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()