# 🖋 Sindhi Alphabet Recognizer

A deep learning-powered web app that classifies **handwritten Sindhi alphabets** using a **CNN model** with **EfficientNetV2** feature extractor and a user-friendly **Streamlit** interface.

## 📁 Project Structure

| Folder/File                | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `dataset/`                 | Contains handwritten Sindhi alphabet dataset (56 folders) |
| `notebook/`                | Contains training notebook                                |
| `notebook/training.ipynb`  | Jupyter notebook for training the CNN model               |
| `test_results/`            | Sample prediction result images                           |
| `model/`                   | Contains the saved trained model                          |
| `model/sindhi_model.keras` | Trained TensorFlow model                                  |
| `main.py`                  | Streamlit app for running the prediction                  |
| `requirements.txt`         | Python package dependencies                               |
| `README.md`                | Project documentation                                     |



---

## 📦 Dataset

- The dataset consists of **56 classes** of Sindhi characters, each containing **58 images**.
- Image format: `.png` or `.jpg`, size varies.
- Total images: **3,016**
- Source: [Kaggle - Sindhi Alphabets Dataset](https://www.kaggle.com/datasets/mudasirmurtaza/sindhi-alphabets)

---

## 🧠 Model Details

- **Framework:** TensorFlow / Keras
- **Base Model:** EfficientNetV2 B0 or B3 from [TensorFlow Hub](https://tfhub.dev/)
- **Input Shape:** `(224, 224, 3)`
- **Architecture:**
  - Data Augmentation
  - EfficientNetV2 (Frozen)
  - Dense(256) + ReLU
  - BatchNormalization + Dropout(0.3)
  - Dense(52) with Softmax activation


## 🚀 How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/Faizan-Ali-Memon/SindhiAlphabetRecognizer.git
cd SindhiAlphabetRecognizer


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ Make sure you are using Python 3.8+ and TensorFlow ≥ 2.10

### 3. Launch the App

```bash
streamlit run main.py
```

### 4. Use the Interface

- Upload an image of a handwritten Sindhi character.
- The app will display the image and predict its class.

---

## 🧪 Training the Model

1. Open the training notebook:

```bash
cd notebook/
jupyter notebook training.ipynb
```

2. Modify paths if needed and run the notebook.
3. Model will be saved at `model/sindhi_model.keras`
4. Class labels will be saved to `model/classes.txt`

---

## 🖼️ Sample Output

**Input:** Handwritten image of a Sindhi character  
**Output:** `Predicted: Class 18 — Sindhi letter ڄ`

---

## 🛠️ Technologies Used

- Python 3.9+
- TensorFlow / Keras
- TensorFlow Hub
- NumPy, OpenCV, Pillow
- Streamlit

---

## 📌 Notes

- Uses `@st.cache_resource` for efficient model loading
- Avoid `Lambda` layers unless deserialization safety is handled
- Input preprocessing includes:
  - Resizing
  - Normalization
  - RGB channel correction

---