# FAULTX – Predictive Maintenance ML System

An end-to-end **Predictive Maintenance Web Application** built using  
**Machine Learning**, **Flask**, **Firebase Authentication**,  
and a complete **HTML/CSS/JS** frontend.

The system predicts **machine failure**, classifies **failure types**, and  
estimates **Remaining Useful Life (RUL)** using sensor data.

---

## 🚀 Key Features

### 🧠 Machine Learning  
- Engineered a predictive ML model using **22 sensor features**.  
- Achieved:  
  - **92% accuracy** in binary failure detection using Random Forest.  
  - **88% accuracy** in classifying **4 failure types**.  
  - RUL (Remaining Useful Life) estimation.  
- Improved model generalization by **12%** using:  
  - Stratified sampling  
  - 5% sensor noise simulation  
  - Cross-validation  
  - Leakage-free preprocessing  

### 🌐 Web Application  
- Complete UI workflow built using **HTML, CSS, JavaScript**.  
- Backend developed with **Flask**.  
- Integrated:  
  - **Firebase Authentication** (Login/Signup)  
  - **Firestore Database** (Prediction history storage)  
  - ML inference API  

---

## 📁 Project Structure

```
FAULTX/
│── templates/
│     ├── index.html
│     ├── login.html
│     ├── input.html
│     ├── results.html
│     ├── history.html
│     └── seemore.html
│── app.py
│── train.py
│── predictive_maintenance_dataset.csv
│── predictive_maintenance_models.pkl
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Sujitha-1005/FaultX-Predictive-Maintenance.git
cd FaultX-Predictive-Maintenance
```

### 2️⃣ Create a Virtual Environment  
```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Your app will run at:

```
http://127.0.0.1:5000/
```

---

## 📦 Requirements

```
Flask==3.0.2
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.0
matplotlib==3.8.3
seaborn==0.13.2
joblib==1.3.2
requests==2.31.0
pytest==7.4.0
```

---

## 🧪 Model Training

To retrain the ML models:

```bash
python train.py
```

This generates:

- `predictive_maintenance_models.pkl` (Model file)  
- Updated dataset preprocessing  
- Updated performance metrics  

---

## 🔥 Firebase Integration

Features included:

- User Authentication (Login / Signup)  
- Firestore Database for:  
  - Prediction history  
  - Timestamp logging  
  - Sensor input tracking  

Add your Firebase keys in `app.py`:

```python
firebaseConfig = {
  "apiKey": "YOUR_API_KEY",
  "authDomain": "YOUR_PROJECT.firebaseapp.com",
  "projectId": "YOUR_PROJECT",
  "storageBucket": "YOUR_PROJECT.appspot.com",
  "messagingSenderId": "YOUR_SENDER_ID",
  "appId": "YOUR_APP_ID"
}
```

---

## 🤝 Contributing

Pull requests and improvements are welcome!

---

## 📜 License

This project is under the **MIT License**.

