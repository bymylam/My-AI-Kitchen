# 🌮 My AI Kitchen

An AI-powered web app that identifies food from images or text and generates recipes instantly using deep learning and OpenAI.

---

## ✨ Features

* 📷 Upload an image of food and detect what it is using a pretrained AI model (MobileNetV2)
* ⌨️ Or type a food name manually
* 🍳 Generate full recipes (description, ingredients, and steps) using OpenAI GPT
* ⚡ Fast, lightweight Streamlit web interface

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **AI Model:** TensorFlow (MobileNetV2 pretrained on ImageNet)
* **Image Processing:** OpenCV, Pillow
* **Recipe Generation:** OpenAI GPT-4o-mini
* **Environment Management:** python-dotenv / Streamlit secrets

---

## 📁 Project Structure

```
MyAIKitchen/
│── app.py                  # Main Streamlit app
│── requirements.txt        # Dependencies
│── .env                    # (optional) API keys
│── .streamlit/secrets.toml # Streamlit secrets (recommended)
│── README.md               # Project documentation
```

---

## 🚀 How It Works

### 1. Image Classification

* User uploads an image
* Image is resized to 224x224
* Preprocessed using `preprocess_input`
* Passed into MobileNetV2 model
* Model returns top 3 predictions

### 2. Recipe Generation

* The top prediction (food label) is sent to OpenAI
* GPT returns:

  * Short description
  * Ingredients
  * Step-by-step instructions

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MyAIKitchen.git
cd MyAIKitchen
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API key

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your_openai_api_key"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 Example Use Cases

* Upload a picture of pizza → get pizza recipe
* Upload a picture of sushi → get sushi recipe
* Type “fried rice” → get recipe instantly

---

## 🧩 Model Info

This project uses:

* **MobileNetV2** (ImageNet pretrained model with 1000 classes)
* Lightweight and optimized for fast inference

---

## 🔐 Environment Variables

Required:

```
OPENAI_API_KEY
```

---

## 📌 Future Improvements

* Add multilingual recipe output
* Improve food detection accuracy with custom dataset
* Add calorie/nutrition estimation
* Save favorite recipes

---

## 👩🏻‍💻 Author

**My Lam**

---

## 📜 License

This project is open-source for learning and personal use.
