# 🎭 Emotion Classification System

<div align="center">

![Emotion Classification](https://github.com/Djaberboudaoud/Emotion-Classification-System/blob/main/result.png)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)

</div>

## 📖 Overview

This project implements a deep learning system for recognizing human emotions from text using LSTM neural networks. The model classifies text into six emotion categories: **joy**, **sadness**, **love**, **anger**, **fear**, and **surprise**.

## 🚀 Features

- **Text-based Emotion Detection**: Classifies emotions from short text sentences
- **Deep Learning Model**: LSTM architecture with embedding layers
- **Real-time Prediction**: Instant emotion analysis through web interface
- **User-friendly Interface**: Built with Streamlit for easy interaction
- **High Accuracy**: Trained on 20,000 labeled text samples

## 🎯 Emotion Categories

| Emotion | Icon | Description |
|---------|------|-------------|
| Joy | 😃 | Happiness, excitement, pleasure |
| Sadness | 😢 | Sorrow, disappointment, grief |
| Love | ❤️ | Affection, romance, care |
| Anger | 😡 | Fury, irritation, rage |
| Fear | 😨 | Anxiety, terror, worry |
| Surprise | 😲 | Astonishment, amazement |


### Model Components
- **Embedding Layer**: 128-dimensional word embeddings
- **LSTM Layer**: 128 units with dropout regularization
- **Dense Layers**: Fully connected layers with ReLU activation
- **Output Layer**: Softmax activation for 6-class classification

## 📊 Dataset

The model is trained on the **Emotions Dataset for NLP** from Kaggle:
- **20,000 text samples** across 6 emotion categories
- **Training**: 16,000 samples
- **Validation**: 2,000 samples  
- **Test**: 2,000 samples

## 🏗️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/Djaberboudaoud/Emotion-Classification-System.git
cd Emotion-Classification-System
