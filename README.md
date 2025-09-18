# 🧠 DeepFake Detector


A deep learning–powered web application to detect **real** vs **fake (AI-generated)** faces using a **MobileNetV2** model. Built with **TensorFlow**, served via **Flask**, and deployed from Google Colab to GitHub.

![App Screenshot](./static/demo_screenshot.png) <!-- Optional: Replace or remove -->

---

## 📁 Dataset

Trained on [140K Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

- **Real:** CelebA dataset
- **Fake:** AI-generated faces (StyleGAN2)

---

## ⚙️ Features

- 🔍 Upload an image and detect whether it's Real or Fake
- ✅ Shows a confidence score (%)
- 🧠 Powered by fine-tuned MobileNetV2
- 📷 Optional face detection using OpenCV before classification
- 📊 Trained in Google Colab with GPU acceleration
- 🌐 Flask-based local web app

---

## 🧪 Model Training (Colab)

Trained using Google Colab with:
- `MobileNetV2` as base model (`include_top=False`)
- Data augmentation with `ImageDataGenerator`
- Weighted loss for class imbalance
- Model saved as `deepfake_detector.keras` and pushed to GitHub

👉 **Model location**: `/model/deepfake_detector.keras`

---

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/ishan1904/deep-fake-detector.git
cd deep-fake-detector
```
 ### 2. Create Virtual Environment
 ```bash
python -m venv .venv
source .venv/bin/activate  # on Linux/Mac
.venv\Scripts\activate      # on Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the App
```bash
python app.py
```
Open browser at: http://127.0.0.1:5000/ 

