# 🏭 Manufacturing Equipment Output Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning system that predicts **Parts Per Hour** (machine output) using manufacturing machine parameters. Built with **Linear Regression**, **FastAPI** backend, and **Streamlit** frontend.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [Training the Model](#-training-the-model)
- [Running Locally](#-running-locally)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting manufacturing equipment output. The system uses Linear Regression to predict **Parts Per Hour** based on various manufacturing parameters including:

- Machine parameters (temperature, pressure, cycle time)
- Environmental factors (ambient temperature)
- Operator information (experience, shift)
- Process metrics (efficiency score, machine utilization)

---

## 🏗 Project Architecture

```
┌─────────────────┐     HTTP/REST      ┌─────────────────┐
│                 │ ◄───────────────► │                 │
│    Frontend     │                    │     Backend     │
│   (Streamlit)   │     JSON Data      │    (FastAPI)    │
│                 │ ◄───────────────► │                 │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                │ Loads
                                                ▼
                                       ┌─────────────────┐
                                       │                 │
                                       │   ML Model      │
                                       │  (model.pkl)    │
                                       │  (scaler.pkl)   │
                                       │                 │
                                       └─────────────────┘
```

---

## ✨ Features

- **Machine Learning Model**: Linear Regression trained on manufacturing data
- **REST API**: FastAPI backend with automatic documentation
- **Web Interface**: User-friendly Streamlit frontend
- **Real-time Predictions**: Instant predictions based on input parameters
- **Scalable Architecture**: Separate frontend and backend services
- **Production Ready**: Configured for deployment on Render

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|------------|
| ML Model | Scikit-learn (Linear Regression) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Deployment | Render |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/manufacturing-output-prediction.git
cd manufacturing-output-prediction
```

---

## 📊 Training the Model

### Option 1: Google Colab (Recommended)

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)

2. Upload the training script and dataset:
   - Upload `training/train.py`
   - Upload `dataset/manufacturing_dataset_1000_samples.csv`

3. Install required packages:
   ```python
   !pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

4. Run the training script:
   ```python
   %run train.py
   ```

5. Download the generated model files:
   - `model.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`
   - `feature_names.pkl`

6. Place the downloaded files in the `model/` folder

### Option 2: Local Training

```bash
cd training
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python train.py
mv *.pkl ../model/
```

---

## 💻 Running Locally

### Step 1: Set Up Backend

```bash
# Navigate to backend folder
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Step 2: Set Up Frontend

Open a new terminal:

```bash
# Navigate to frontend folder
cd frontend

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the frontend
streamlit run app.py
```

Frontend will be available at: `http://localhost:8501`

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | API health status |
| POST | `/predict` | Make prediction |
| GET | `/model/info` | Get model information |

### Prediction Request Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Injection_Temperature": 220.0,
    "Injection_Pressure": 130.0,
    "Cycle_Time": 30.0,
    "Cooling_Time": 12.0,
    "Material_Viscosity": 300.0,
    "Ambient_Temperature": 25.0,
    "Machine_Age": 5.0,
    "Operator_Experience": 10.0,
    "Maintenance_Hours": 50.0,
    "Shift": "Day",
    "Machine_Type": "Type_A",
    "Material_Grade": "Standard",
    "Day_of_Week": "Monday",
    "Temperature_Pressure_Ratio": 1.7,
    "Total_Cycle_Time": 42.0,
    "Efficiency_Score": 0.5,
    "Machine_Utilization": 0.45
  }'
```

### Response Example

```json
{
  "success": true,
  "predicted_parts_per_hour": 42.56,
  "message": "Prediction successful"
}
```

---

## 🌐 Deployment

### Deploy to Render

#### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Manufacturing Output Prediction"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/manufacturing-output-prediction.git
git push -u origin main
```

#### Step 2: Deploy Backend on Render

1. Go to [render.com](https://render.com) and sign in
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `manufacturing-prediction-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click "Create Web Service"

#### Step 3: Deploy Frontend on Render

1. Click "New" → "Web Service"
2. Connect the same repository
3. Configure:
   - **Name**: `manufacturing-prediction-frontend`
   - **Root Directory**: `frontend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add Environment Variable:
   - `BACKEND_URL`: Your backend Render URL
5. Click "Create Web Service"

#### Step 4: Update Frontend Backend URL

After deploying, update the `BACKEND_URL` in `frontend/app.py` with your actual Render backend URL.

---

## 📁 Project Structure

```
manufacturing_output_prediction/
│
├── frontend/                    # Streamlit frontend application
│   ├── app.py                   # Main Streamlit app
│   └── requirements.txt         # Frontend dependencies
│
├── backend/                     # FastAPI backend application
│   ├── main.py                  # API entry point
│   ├── predict.py               # Prediction logic
│   ├── preprocess.py            # Data preprocessing
│   └── requirements.txt         # Backend dependencies
│
├── model/                       # Trained model files
│   ├── model.pkl                # Linear Regression model
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoders.pkl       # Categorical encoders
│   └── feature_names.pkl        # Feature names
│
├── training/                    # Model training scripts
│   └── train.py                 # Training script
│
├── dataset/                     # Dataset files
│   └── manufacturing_dataset_1000_samples.csv
│
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

---

## 📈 Model Performance

| Metric | Training | Test |
|--------|----------|------|
| R² Score | ~0.95 | ~0.93 |
| RMSE | ~3.5 | ~4.2 |
| MAE | ~2.8 | ~3.3 |

*Note: Actual metrics may vary based on training run*

---

## 🔗 Deployment Links

| Service | URL |
|---------|-----|
| Frontend | [Your Streamlit URL](https://your-frontend.onrender.com) |
| Backend | [Your FastAPI URL](https://your-backend.onrender.com) |
| API Docs | [Your API Docs](https://your-backend.onrender.com/docs) |

*Update links after deployment*

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)

---

## ⭐ Acknowledgments

- Dataset: Manufacturing sensor data
- Built with FastAPI and Streamlit
- Deployed on Render

---

*Made with ❤️ for Manufacturing Intelligence*
