🌱 Soybean Variety Analyzer


<div align="center">
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/ML-RandomForest-green
https://img.shields.io/badge/GUI-PyQt5-orange
https://img.shields.io/badge/Computer%2520Vision-OpenCV-red

A smart desktop application that helps farmers and researchers identify soybean varieties through image analysis

</div>

✨ What Does It Do?
Simply upload a photo of soybeans, and our AI-powered system will:

🔍 Identify the variety (1110, 1135, 2172, etc.)

📊 Analyze plant health based on color patterns

🌿 Estimate growth stage (early, mid, mature)

💡 Provide recommendations for care and maintenance

🛠️ Tech Stack
Core Technologies
Python 3.8+ - Main programming language

PyQt5 - Beautiful desktop GUI framework

OpenCV - Image processing and analysis

scikit-learn - Machine learning algorithms

pandas & NumPy - Data processing and analysis

Machine Learning
Random Forest Classifier - Main prediction model

SMOTE - Handling class imbalance

Feature Engineering - Advanced RGB-based features

Cross-Validation - Reliable accuracy testing

Data Processing
Excel Integration - Works with your existing data

Data Augmentation - Enhanced training datasets

Real-time Analysis - Instant results

🚀 How It Works
1. Take a photo of your soybean plants

2. Upload the image through the simple interface

3. Get instant analysis including:

i. Variety identification with confidence scores

ii. Growth stage estimation

iii. Health assessment

Care recommendations

📦 Installation
# Clone the repository
git clone https://github.com/akshatrsingh21/soybean-analyzer.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python soybean_analyzer.py

🎯 Key Features
1. User-Friendly Interface - Drag and drop image upload

2. Real-time Processing - Results in seconds

3. Accuracy Metrics - Confidence scores for predictions

4. Export Results - Save analysis reports

5. Multi-Platform - Works on Windows, macOS, and Linux
   

📊 Sample Results

🌱 SOYBEAN ANALYSIS REPORT
========================================
📁 Image: soybean_field.jpg
📏 Size: 4000x3000 pixels

🎨 COLOR ANALYSIS:
   RGB Values: R=128, G=107, B=67
   Brightness: 100.7

🌱 VARIETY IDENTIFICATION:
   2172 - 92.3% confidence

📈 GROWTH ANALYSIS:
   Growth Stage: Mature Stage (Day 15+)
   Health Status: Excellent Health

💡 RECOMMENDATIONS:
   • Continue current care routine
   • Monitor for harvest readiness
🤝 Contributing


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


🙋‍♂️ Need Help?
Have questions or need support?

Open an issue on GitHub

Check the documentation

Reach out to our community

<div align="center">

  
Built with ❤️ for the agricultural community



</div>


🎨 Project Structure

soybean-analyzer/
├── data/ # Training datasets
├── models/ # Trained ML models
├── src/ # Source code
│ ├── gui.py # PyQt5 interface
│ ├── ml_model.py # Machine learning core
│ └── image_processor.py # OpenCV operations
├── docs/ # Documentation
├── requirements.txt # Dependencies
└── README.md # This file

🌟 Why This Project Matters
This tool bridges the gap between traditional farming and modern technology, making AI-powered plant analysis accessible to everyone in the agricultural sector. By automating variety identification, we help farmers save time and make better decisions about crop management.



