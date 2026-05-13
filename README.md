# Smart_Crop_Disease_Detection_Syste 🌱

An AI-powered edge computing system that detects crop diseases from leaf images using a deep learning model deployed on Raspberry Pi 4.

---

## 📌 Overview

Crop diseases cause significant agricultural losses worldwide. This project combines Deep Learning, Computer Vision, and Edge AI to create a real-time, offline-capable crop disease detection system.

The system captures leaf images using a Raspberry Pi Camera Module, processes them locally using a TensorFlow Lite optimized MobileNetV2 model, and predicts both the plant species and disease class.

Environmental conditions such as temperature and humidity are also monitored using a DHT22 sensor to provide additional disease-related insights.

---

## 🚀 Features

* 🌿 Real-time crop disease detection
* 📷 Leaf image classification using Deep Learning
* 🧠 MobileNetV2-based hierarchical architecture
* ⚡ TensorFlow Lite optimized edge inference
* 💻 Raspberry Pi 4 deployment
* 🌡️ Temperature and humidity monitoring using DHT22 sensor
* 📴 Fully offline functionality
* 📊 38 disease classes across 14 crop species
* ✅ Validation Accuracy: **93.58%**

---

## 🛠️ Tech Stack

### Software

* Python
* TensorFlow
* TensorFlow Lite
* OpenCV
* NumPy
* Pandas
* Picamera2

### Hardware

* Raspberry Pi 4
* Raspberry Pi Camera Module V2
* DHT22 Sensor
* TFT LCD Display

---

## 🧠 Model Architecture

The system uses a custom **Hierarchical Two-Branch Architecture** built on top of **MobileNetV2**.

### Branches:

1. **Plant Classification Branch**

   * Identifies the crop species.

2. **Disease Classification Branch**

   * Predicts the disease using image features combined with plant classification output.

This architecture ensures biologically consistent predictions by preventing diseases from being predicted for the wrong plant species.

---

## 📂 Dataset

The model was trained using the **PlantVillage Dataset** containing:

* 54,000+ images
* 38 disease classes
* 14 crop species

### Supported Crops

* Tomato
* Potato
* Corn
* Apple
* Grape
* Pepper
* Strawberry
* Peach
* Cherry
* Blueberry
* Orange
* Raspberry
* Soybean
* Squash

---

## 📈 Results

| Metric                        | Value  |
| ----------------------------- | ------ |
| Validation Accuracy           | 93.58% |
| Plant Classification Accuracy | 97.56% |
| Overall Accuracy              | 94%    |
| Model Size Reduction          | ~70%   |

### Best Performing Classes

* Apple Cedar Apple Rust
* Orange Huanglongbing
* Grape Leaf Blight
* Squash Powdery Mildew
* Tomato Yellow Leaf Curl Virus

---

## ⚙️ Edge Deployment

The trained `.keras` model was converted into a lightweight `.tflite` model for efficient edge deployment.

### Deployment Components

* `plant_disease.tflite`
* `class_names.json`
* `plant_names.json`
* `predict.py`

The system performs inference locally on Raspberry Pi without requiring internet connectivity.

---

## 📁 Project Structure

```bash
smart-crop-disease-detection/
│
├── README.md
├── requirements.txt
├── predict.py
├── train.py
│
├── model/
│   ├── plant_disease.tflite
│   ├── class_names.json
│   └── plant_names.json
│
├── images/
│   ├── output1.png
│   ├── output2.png
│   └── architecture.png
│
├── notebooks/
│   └── training_notebook.ipynb
│
└── report/
    └── project_report.pdf
```

---

## 🔧 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/smart-crop-disease-detection.git

cd smart-crop-disease-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Prediction

```bash
python predict.py
```

### Input

* Capture image using Raspberry Pi Camera
* OR provide a stored leaf image

### Output

* Plant species
* Disease prediction
* Confidence score
* Temperature and humidity readings

## 🌍 Real-World Impact

This system helps farmers by:

* Detecting diseases early
* Reducing crop losses
* Lowering pesticide usage
* Enabling targeted treatment
* Providing affordable AI-based agricultural assistance

---

## 🔮 Future Improvements

* Mobile application integration
* Cloud-based monitoring dashboard
* Multi-language support
* Real-time field-wide monitoring
* Advanced disease severity estimation

---

## 👨‍💻 Contributors

* Hanusha K

## 📜 License

This project is created for educational and research purposes.

---

## ⭐ Acknowledgements

* PlantVillage Dataset
* TensorFlow Team
* Raspberry Pi Foundation
* Open-source AI co
