import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import cv2
import numpy as np
import pandas as pd
import joblib

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image file: {image_path}")

    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_rgb = avg_color[::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
    else:
        area = 0
        length = 0

    features = {
        'Average R': avg_color_rgb[0],
        'Average G': avg_color_rgb[1],
        'Average B': avg_color_rgb[2],
        'Area': area,
        'Length': length,
        'Weight': 0  # Placeholder
    }
    return features


class SoybeanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Soybean Seed Varity Predictor")
        self.geometry("600x400")

        self.model = joblib.load('soybean_model.joblib')
        self.feature_cols = joblib.load('feature_columns.joblib')

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Select a Soybean Seed Image to Predict Variety", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(self, text="Browse Image", command=self.browse_image)
        self.upload_button.pack(pady=5)

        self.result_text = ScrolledText(self, height=15)
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def browse_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Soybean Seed Image",
            filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*"))
        )
        if image_path:
            try:
                features = extract_features(image_path)
                self.show_result(features)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def show_result(self, features):
        self.result_text.delete('1.0', tk.END)
        # Display extracted features
        self.result_text.insert(tk.END, "Extracted Features:\n")
        for k, v in features.items():
            self.result_text.insert(tk.END, f"{k}: {v:.2f}\n")

        # Prepare dataframe for prediction
        df_features = pd.DataFrame([features])
        df_features = df_features[self.feature_cols].fillna(df_features.median())

        # Predict variety
        prediction = self.model.predict(df_features)

        self.result_text.insert(tk.END, f"\nPredicted Soybean Variety: {prediction[0]}\n")


if __name__ == "__main__":
    app = SoybeanApp()
    app.mainloop()
