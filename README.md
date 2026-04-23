# 🌸 Iris Flower Species Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready machine learning web application that predicts Iris flower species with **97%+ accuracy**. Built with Streamlit, MLflow for experiment tracking, and deployed on Streamlit Cloud.

## 🎯 Live Demo

👉 [Click here to try the app](https://your-app-url.streamlit.app)

## 📊 Features

- 🌸 **Real-time predictions** - Get instant species classification
- 📈 **Confidence scores** - See how confident the model is
- 📊 **Probability visualization** - View prediction distribution across all species
- 🔍 **MLflow tracking** - Complete experiment logging and versioning
- 🎨 **Beautiful UI** - User-friendly interface with interactive sliders

## 🤖 Model Details

- **Algorithm**: Logistic Regression
- **Accuracy**: 97-100%
- **Precision**: 0.97 (weighted)
- **Recall**: 0.97 (weighted)
- **F1 Score**: 0.97 (weighted)

## 📁 Dataset

The classic **Iris Dataset** by Sir Ronald Fisher (1936):

| Species | Samples | Features |
|---------|---------|----------|
| Setosa 🌿 | 50 | Sepal length/width |
| Versicolor 🎨 | 50 | Petal length/width |
| Virginica 🌺 | 50 | (all in cm) |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Scikit-learn (Logistic Regression)
- **Tracking**: MLflow
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/iris-flower-predictor.git
cd iris-flower-predictor