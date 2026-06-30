# 🌦️ AI-Powered Real-Time Weather Forecasting Dashboard

An AI-powered weather dashboard built using **Streamlit**, **Python**, and **Machine Learning** that provides real-time weather information, detects unusual weather conditions, and recommends cities with similar weather patterns.

## 📌 Project Overview

The **AI-Powered Real-Time Weather Forecasting Dashboard** is an interactive web application that fetches live weather data from the **OpenWeatherMap API** and applies **Machine Learning** techniques to generate intelligent insights.

Unlike traditional weather applications, this dashboard not only displays weather information but also performs **weather anomaly detection** using the **Isolation Forest** algorithm and recommends cities with similar weather conditions.

This project was developed as part of the **Fundamentals of Machine Learning (FML)** course and is suitable for **Smart India Hackathon (SIH)** under the theme **AI for Weather and Climate Awareness**.

---

## 🚀 Features

- 🌍 Real-time weather data
- 🌡 Temperature, humidity, and wind speed
- 📊 Interactive Plotly visualizations
- 🗺 Weather location displayed on an interactive map
- 🤖 AI-based Weather Anomaly Detection using Isolation Forest
- 🌏 Smart Weather Recommendation System
- 📱 User-friendly Streamlit dashboard
- ⚡ Fast and responsive interface

---

## 🧠 Machine Learning Concepts Used

### 1. Isolation Forest (Anomaly Detection)

The project uses the **Isolation Forest** algorithm from Scikit-Learn to identify unusual weather conditions.

It analyzes weather parameters such as:

- Temperature
- Humidity
- Wind Speed

and classifies observations as:

- ✅ Normal
- ⚠️ Anomaly

This helps identify abnormal weather behavior.

---

### 2. Smart Weather Recommendation

The system compares the current city's weather with multiple Indian cities using similarity scoring based on:

- Temperature
- Humidity

It recommends cities having weather conditions similar to the selected location.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Dashboard |
| OpenWeatherMap API | Real-time Weather Data |
| Scikit-Learn | Machine Learning |
| Isolation Forest | Anomaly Detection |
| Plotly | Interactive Charts |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Requests | API Integration |

---

## 📂 Project Structure

```
AI-Weather-Dashboard/
│
├── weather_dashboard.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Weather-Dashboard.git
```

Move into the project folder

```bash
cd AI-Weather-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run weather_dashboard.py
```

---

## 📦 Requirements

```
streamlit
plotly
pandas
numpy
requests
scikit-learn
```

---

## 🔑 API Setup

Create a free API key from:

https://openweathermap.org/api

Replace the API key inside the code:

```python
API_KEY = "API_KEY"
```

---

## 📊 Dashboard Modules

### 🌡 Weather Information

- Temperature
- Humidity
- Wind Speed
- Weather Condition

---

### 📈 Weather Visualization

- Interactive Bar Chart
- Weather Comparison
- Geo Map

---

### 🤖 AI Weather Analysis

Isolation Forest detects:

- Extreme Temperature
- Abnormal Humidity
- Unusual Wind Speed

---

### 🌏 Smart Recommendation

Suggests top cities having weather conditions similar to the searched location.

---

## 🎯 Project Outcomes

- Real-time weather monitoring
- Intelligent weather anomaly detection
- Smart city recommendation system
- Interactive visualization dashboard
- Practical implementation of Machine Learning

---

## 🔮 Future Enhancements

- 7-Day Weather Forecast
- LSTM-based Time Series Prediction
- Rainfall Prediction
- Air Quality Index (AQI)
- Weather Alerts
- Satellite Data Integration
- Firebase User Authentication
- Mobile Application

---

---



Theme:
> Tech for Social Good – AI for Weather and Climate Awareness

Course:
**Fundamentals of Machine Learning (FML)**

Department:
**Artificial Intelligence and Data Science**

---

## 👨‍💻 Author

**Pushpak Bala Gaidhane**
