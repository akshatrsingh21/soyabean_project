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
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage import feature, filters, morphology
import joblib
import warnings
warnings.filterwarnings('ignore')

# Pre-trained variety detection model (simulated - in real application, you'd train this on your data)
class VarietyDetector:
    def __init__(self):
        # This would be replaced with your actual trained model
        self.model = self.create_model()
        self.scaler = StandardScaler()
        
    def create_model(self):
        # Simulate a trained model - in practice, train this on your soybean data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model
        
    def extract_features(self, image):
        """Extract comprehensive features from soybean image"""
        features = {}
        
        # Color features
        features['avg_r'] = np.mean(image[:,:,0])
        features['avg_g'] = np.mean(image[:,:,1])
        features['avg_b'] = np.mean(image[:,:,2])
        features['std_r'] = np.std(image[:,:,0])
        features['std_g'] = np.std(image[:,:,1])
        features['std_b'] = np.std(image[:,:,2])
        
        # Texture features using GLCM
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        glcm = feature.graycomatrix(gray, [1], [0], symmetric=True, normed=True)
        features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Shape features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return np.array(list(features.values())).reshape(1, -1)
    
    def predict_variety(self, image):
        """Predict soybean variety using comprehensive features"""
        features = self.extract_features(image)
        
        # Based on extensive analysis of your dataset patterns:
        avg_r, avg_g, avg_b = features[0, 0], features[0, 1], features[0, 2]
        
        # Variety 2172: High RGB values (mature plants)
        if avg_r > 100 and avg_g > 80 and avg_b > 45:
            confidence = min(100, ((avg_r - 100) + (avg_g - 80)) / 2)
            return f"2172 (Mature Variety) - {confidence:.1f}% confidence"
        
        # Variety 1135: Medium values, specific color balance
        elif 30 <= avg_r <= 50 and 32 <= avg_g <= 44 and 37 <= avg_b <= 49:
            return "1135 (Medium Growth Variety)"
        
        # Variety 1110: Lower values, specific pattern
        elif avg_r < 40 and avg_g < 42 and avg_b < 47:
            return "1110 (Early Growth Variety)"
        
        # Fallback with detailed analysis
        else:
            return self.detailed_variety_analysis(avg_r, avg_g, avg_b)
    
    def detailed_variety_analysis(self, r, g, b):
        """Detailed analysis for uncertain cases"""
        # Calculate similarity scores to known variety profiles
        score_1110 = 100 - (abs(r-35) + abs(g-34) + abs(b-40)) / 3
        score_1135 = 100 - (abs(r-37) + abs(g-37) + abs(b-43)) / 3
        score_2172 = 100 - (abs(r-128) + abs(g-107) + abs(b-67)) / 3
        
        scores = {
            '1110': max(0, score_1110),
            '1135': max(0, score_1135),
            '2172': max(0, score_2172)
        }
        
        best_variety = max(scores.items(), key=lambda x: x[1])
        
        if best_variety[1] > 60:
            return f"{best_variety[0]} ({'Early' if best_variety[0]=='1110' else 'Medium' if best_variety[0]=='1135' else 'Mature'}) - {best_variety[1]:.1f}% match"
        else:
            return f"Uncertain - Closest: {best_variety[0]} ({best_variety[1]:.1f}% match)"

class AdvancedRootAnalyzer:
    def __init__(self):
        pass
    
    def analyze_roots(self, image):
        """Advanced root analysis using multiple techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multiple edge detection methods
        edges_canny = cv2.Canny(gray, 30, 100)
        edges_sobel = filters.sobel(gray) > 0.05
        edges_laplace = np.abs(cv2.Laplacian(gray, cv2.CV_64F)) > 0.5
        
        # Combine edge detections
        combined_edges = edges_canny | edges_sobel | edges_laplace
        
        # Morphological operations to clean up
        cleaned_edges = morphology.remove_small_objects(combined_edges, min_size=20)
        cleaned_edges = morphology.binary_closing(cleaned_edges, morphology.disk(2))
        
        # Calculate root metrics
        root_pixels = np.sum(cleaned_edges)
        total_pixels = cleaned_edges.size
        root_density = (root_pixels / total_pixels) * 100
        
        # Root complexity (branching)
        skeleton = morphology.skeletonize(cleaned_edges)
        branch_points = self.detect_branch_points(skeleton)
        
        return {
            'density': root_density,
            'complexity': len(branch_points) / max(1, root_pixels) * 1000,
            'total_roots': root_pixels
        }
    
    def detect_branch_points(self, skeleton):
        """Detect branch points in skeletonized image"""
        kernel = np.ones((3,3), np.uint8)
        intersections = cv2.morphologyEx(skeleton.astype(np.uint8), cv2.MORPH_HITMISS, np.array([[1,1,1],[1,1,1],[1,1,1]]))
        return np.where(intersections > 0)

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.variety_detector = VarietyDetector()
        self.root_analyzer = AdvancedRootAnalyzer()

    def run(self):
        try:
            self.progress_signal.emit(10)
            
            # Load and preprocess image
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.progress_signal.emit(30)
            
            # Enhanced color analysis
            color_results = self.analyze_colors(image_rgb)
            self.progress_signal.emit(50)
            
            # Advanced variety detection
            variety = self.variety_detector.predict_variety(image_rgb)
            self.progress_signal.emit(70)
            
            # Advanced root analysis
            root_results = self.root_analyzer.analyze_roots(image_rgb)
            self.progress_signal.emit(90)
            
            # Comprehensive health assessment
            health_status = self.assess_health(color_results, root_results)
            growth_stage = self.estimate_growth_stage(color_results, root_results)
            
            results = {
                'image_path': self.image_path,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'color_analysis': color_results,
                'variety': variety,
                'root_analysis': root_results,
                'health_status': health_status,
                'growth_stage': growth_stage,
                'recommendations': self.generate_recommendations(health_status, root_results)
            }
            
            self.progress_signal.emit(100)
            self.result_signal.emit(results)
            
        except Exception as e:
            self.result_signal.emit({'error': str(e)})
        finally:
            self.finished_signal.emit()

    def analyze_colors(self, image):
        """Comprehensive color analysis"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        return {
            'rgb_mean': [round(np.mean(image[:,:,i]), 2) for i in range(3)],
            'rgb_std': [round(np.std(image[:,:,i]), 2) for i in range(3)],
            'hsv_mean': [round(np.mean(hsv[:,:,i]), 2) for i in range(3)],
            'lab_mean': [round(np.mean(lab[:,:,i]), 2) for i in range(3)],
            'brightness': round(np.mean(image), 2),
            'green_dominance': round(np.mean(image[:,:,1]) / max(1, np.mean(image[:,:,0])), 2)
        }

    def assess_health(self, color_results, root_results):
        """Comprehensive health assessment"""
        r, g, b = color_results['rgb_mean']
        root_density = root_results['density']
        green_dominance = color_results['green_dominance']
        
        health_score = 0
        
        # Green dominance is good
        if green_dominance > 1.3:
            health_score += 30
        elif green_dominance > 1.1:
            health_score += 20
        elif green_dominance > 0.9:
            health_score += 10
            
        # Root development
        if root_density > 2.0:
            health_score += 40
        elif root_density > 1.0:
            health_score += 25
        elif root_density > 0.5:
            health_score += 10
            
        # Color balance
        if g > max(r, b) * 1.2:
            health_score += 30
            
        if health_score >= 80:
            return "Excellent Health"
        elif health_score >= 60:
            return "Good Health"
        elif health_score >= 40:
            return "Fair Health - Monitor"
        elif health_score >= 20:
            return "Poor Health - Needs Attention"
        else:
            return "Critical - Immediate Action Needed"

    def estimate_growth_stage(self, color_results, root_results):
        """Improved growth stage estimation"""
        brightness = color_results['brightness']
        root_density = root_results['density']
        green_dominance = color_results['green_dominance']
        
        if brightness < 40 and root_density < 0.3:
            return "Germination Stage (Day 1-3)"
        elif brightness < 60 and root_density < 1.0:
            return "Early Growth (Day 4-7)"
        elif brightness < 90 and root_density < 2.0:
            return "Vegetative Stage (Day 8-14)"
        elif brightness < 120:
            return "Flowering Stage (Day 15-21)"
        else:
            return "Maturation Stage (Day 22+)"

    def generate_recommendations(self, health_status, root_results):
        """Generate specific recommendations"""
        recommendations = []
        
        if "Excellent" in health_status:
            recommendations.extend([
                "✓ Optimal growth conditions",
                "✓ Continue current regimen",
                "✓ Monitor for pests"
            ])
        elif "Good" in health_status:
            recommendations.extend([
                "✓ Good overall health",
                "● Maintain watering schedule",
                "● Check nutrient levels weekly"
            ])
        elif "Fair" in health_status:
            recommendations.extend([
                "⚠️ Monitor plant closely",
                "💧 Adjust watering if needed",
                "🌱 Consider mild fertilizer"
            ])
        elif "Poor" in health_status:
            recommendations.extend([
                "⚠️ Needs immediate attention",
                "💧 Check soil moisture",
                "🌿 Apply balanced nutrients",
                "🔍 Inspect for diseases"
            ])
        else:
            recommendations.extend([
                "🚨 Critical condition",
                "💧 Urgent water adjustment",
                "🌿 Emergency nutrients needed",
                "🩺 Professional consultation recommended"
            ])
            
        # Root-specific recommendations
        if root_results['density'] < 0.5:
            recommendations.extend([
                "🌱 Root development poor",
                "💧 Improve soil aeration",
                "🌿 Root-stimulating fertilizer needed"
            ])
        elif root_results['density'] < 1.0:
            recommendations.append("● Monitor root development")
            
        return recommendations

class SoybeanAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image_path = None

    def initUI(self):
        self.setWindowTitle("Advanced Soybean Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel
        left_panel = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        self.image_label.setText("No image selected\n\nClick 'Load Image' to begin")
        self.image_label.setFont(QFont("Arial", 12))
        
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("📁 Load Soybean Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 16px; padding: 15px; border-radius: 8px; }")
        
        self.analyze_btn = QPushButton("🔍 Analyze Image")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 16px; padding: 15px; border-radius: 8px; }")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { height: 20px; border-radius: 10px; }")
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.progress_bar)
        control_group.setLayout(control_layout)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(control_group)
        
        # Right panel
        right_panel = QVBoxLayout()
        
        results_group = QGroupBox("📊 Analysis Results")
        results_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: 'Consolas'; font-size: 13px; background-color: #f8f9fa; padding: 10px;")
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        right_panel.addWidget(results_group)
        
        main_layout.addLayout(left_panel, 35)
        main_layout.addLayout(right_panel, 65)

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
            pixmap = pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        self.results_text.append("🌱 ADVANCED SOYBEAN ANALYSIS REPORT")
        self.results_text.append("=" * 50)
        self.results_text.append(f"📁 Image: {os.path.basename(results['image_path'])}")
        self.results_text.append(f"📏 Size: {results['image_size']}")
        self.results_text.append("")
        
        # Color Analysis
        self.results_text.append("🎨 COLOR ANALYSIS:")
        self.results_text.append(f"   RGB Mean: R={results['color_analysis']['rgb_mean'][0]}, G={results['color_analysis']['rgb_mean'][1]}, B={results['color_analysis']['rgb_mean'][2]}")
        self.results_text.append(f"   RGB Std: R={results['color_analysis']['rgb_std'][0]}, G={results['color_analysis']['rgb_std'][1]}, B={results['color_analysis']['rgb_std'][2]}")
        self.results_text.append(f"   Brightness: {results['color_analysis']['brightness']}")
        self.results_text.append(f"   Green Dominance: {results['color_analysis']['green_dominance']}")
        self.results_text.append("")
        
        # Plant Analysis
        self.results_text.append("🌱 PLANT ANALYSIS:")
        self.results_text.append(f"   Variety: {results['variety']}")
        self.results_text.append(f"   Growth Stage: {results['growth_stage']}")
        self.results_text.append(f"   Health Status: {results['health_status']}")
        self.results_text.append("")
        
        # Root Analysis
        self.results_text.append("🌿 ROOT ANALYSIS:")
        self.results_text.append(f"   Density: {results['root_analysis']['density']:.3f}%")
        self.results_text.append(f"   Complexity: {results['root_analysis']['complexity']:.2f}")
        self.results_text.append(f"   Total Root Pixels: {results['root_analysis']['total_roots']}")
        self.results_text.append("")
        
        # Recommendations
        self.results_text.append("💡 RECOMMENDATIONS:")
        for recommendation in results['recommendations']:
            self.results_text.append(f"   {recommendation}")

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