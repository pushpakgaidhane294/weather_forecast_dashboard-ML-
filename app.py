import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import numpy as np
import time

# Streamlit Page Configuration
st.set_page_config(page_title="AI Weather Dashboard", layout="wide")
st.title("🌍 AI-Powered Real-Time Weather Forecasting Dashboard")

st.markdown("""
This dashboard provides *real-time weather insights* 🌦  
and uses **Machine Learning (Isolation Forest + Smart Recommendations)** to:
- Detect unusual weather conditions 🔍  
- Recommend cities with similar weather ☀️🌧️  

Developed for *Smart India Hackathon 2025* 💡  
By Pushpak Bala Gaidhane — Dept. of AI & Data Science
""")

# Sidebar Inputs
st.sidebar.header("⚙ Weather Search Options")
search_type = st.sidebar.radio("Search By:", ["City Name", "ZIP / PIN Code"])
query = st.sidebar.text_input("Enter City or ZIP:", "Delhi")
country = st.sidebar.text_input("Country Code (e.g., in, us, gb):", "in")

API_KEY = "01d1f6ccaffe5c8f36a6196d7c7485a6"

# Fetch Weather Data Function
def get_weather(query, country, by_zip=False):
    try:
        if by_zip:
            url = f"http://api.openweathermap.org/data/2.5/weather?zip={query},{country}&appid={API_KEY}&units=metric"
        else:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={query},{country}&appid={API_KEY}&units=metric"
        
        # Added timeout to prevent hanging
        r = requests.get(url, timeout=5)
        data = r.json()
        
        if r.status_code != 200 or "main" not in data:
            return None, f"❌ Error: {data.get('message', 'Invalid input or API issue')}"
        
        weather_info = {
            "Location": f"{data['name']}, {data['sys']['country']}",
            "Temperature (°C)": data["main"]["temp"],
            "Humidity (%)": data["main"]["humidity"],
            "Wind Speed (km/h)": round(data["wind"]["speed"] * 3.6, 2),
            "Condition": data["weather"][0]["main"],
            "Latitude": data["coord"]["lat"],
            "Longitude": data["coord"]["lon"]
        }
        return weather_info, None
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"🌐 Network error: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Get Weather Button
if st.sidebar.button("🔍 Get Weather Data"):
    with st.spinner("Fetching weather data..."):
        weather_data, error = get_weather(query, country, by_zip=(search_type == "ZIP / PIN Code"))

    if error:
        st.error(error)
        st.stop()
    elif weather_data:
        st.success(f"✅ Weather data fetched for {weather_data['Location']}")
        
        # Show metrics
        st.subheader(f"📍 Current Weather in {weather_data['Location']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature (°C)", f"{weather_data['Temperature (°C)']}°C")
        col2.metric("💧 Humidity (%)", f"{weather_data['Humidity (%)']}%")
        col3.metric("🌬 Wind Speed (km/h)", f"{weather_data['Wind Speed (km/h)']} km/h")
        st.info(f"**Condition:** {weather_data['Condition']}")
        
        
        # 1️⃣ Visualization
        df = pd.DataFrame({
            "Parameter": ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)"],
            "Value": [
                weather_data["Temperature (°C)"],
                weather_data["Humidity (%)"],
                weather_data["Wind Speed (km/h)"]
            ]
        })
        fig_bar = px.bar(df, x="Parameter", y="Value", 
                         title="📊 Weather Parameter Comparison", 
                         color="Parameter", text_auto=True,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_bar, use_container_width=True)

        
        # 2️⃣ Anomaly Detection (ML Concept 1)
        st.subheader("🤖 AI Analysis — Weather Anomaly Detection")
        st.markdown("*Using Isolation Forest ML algorithm to detect unusual weather patterns*")
        
        # Generate synthetic historical data for demonstration
        np.random.seed(42)
        sample_weather = pd.DataFrame({
            "Temperature": np.random.normal(weather_data["Temperature (°C)"], 3, 30),
            "Humidity": np.random.normal(weather_data["Humidity (%)"], 5, 30),
            "Wind": np.random.normal(weather_data["Wind Speed (km/h)"], 2, 30)
        })

        # Train Isolation Forest model
        model = IsolationForest(contamination=0.1, random_state=42)
        preds = model.fit_predict(sample_weather)
        sample_weather["Anomaly"] = np.where(preds == -1, "⚠️ Anomaly", "✅ Normal")

        # 3D Scatter plot
        fig_anomaly = px.scatter_3d(
            sample_weather, x="Temperature", y="Humidity", z="Wind",
            color="Anomaly", title="3D Weather Anomaly Detection",
            color_discrete_map={"⚠️ Anomaly": "red", "✅ Normal": "green"}
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        anomaly_count = (sample_weather["Anomaly"] == "⚠️ Anomaly").sum()
        if anomaly_count > 0:
            st.warning(f"🔍 Detected **{anomaly_count} unusual readings** in the recent data sample.")
        else:
            st.success("✅ All weather readings appear normal!")

        
        # 3️⃣ Smart Recommendations (ML Concept 2)
        st.subheader("🌏 Smart Weather Recommendations (Similar Cities)")
        st.markdown("*Finding cities with similar weather conditions using distance-based similarity*")
        
        cities = ["Mumbai", "Pune", "Chennai", "Bangalore", "Kolkata", "Hyderabad"]
        recs = []
        
        # Progress bar for fetching multiple cities
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, city in enumerate(cities):
            try:
                status_text.text(f"Fetching data for {city}...")
                data, _ = get_weather(city, "in")
                if data:
                    # Calculate similarity score (lower is more similar)
                    diff = abs(data["Temperature (°C)"] - weather_data["Temperature (°C)"]) + \
                           abs(data["Humidity (%)"] - weather_data["Humidity (%)"]) * 0.5
                    recs.append((city, round(diff, 2), data["Temperature (°C)"], data["Humidity (%)"]))
                progress_bar.progress((idx + 1) / len(cities))
                time.sleep(0.2)  # Small delay to avoid rate limiting
            except:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if recs:
            rec_df = pd.DataFrame(recs, columns=["City", "Similarity Score", "Temp (°C)", "Humidity (%)"])
            rec_df = rec_df.sort_values(by="Similarity Score").head(5)
            
            st.dataframe(rec_df, use_container_width=True)
            st.success(f"💡 **{rec_df.iloc[0]['City']}** has the most similar weather to {weather_data['Location']}!")
            
            # Visualization of similar cities
            fig_similarity = px.bar(rec_df, x="City", y="Similarity Score", 
                                   title="City Weather Similarity (Lower = More Similar)",
                                   color="Similarity Score",
                                   color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_similarity, use_container_width=True)
        else:
            st.warning("⚠️ Could not fetch data for comparison cities.")

        
        # Map Visualization
        st.subheader("🗺️ Geographic Location")
        fig_map = go.Figure(go.Scattergeo(
            lon=[weather_data["Longitude"]],
            lat=[weather_data["Latitude"]],
            text=f"{weather_data['Location']}<br>{weather_data['Temperature (°C)']}°C",
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='circle'),
            textposition="top center"
        ))
        fig_map.update_layout(
            title=f"Location: {weather_data['Location']}",
            geo=dict(
                projection_scale=5,
                center=dict(lat=weather_data["Latitude"], lon=weather_data["Longitude"]),
                showland=True,
                landcolor="lightgreen",
                showocean=True,
                oceancolor="lightblue",
                showcountries=True,
                countrycolor="black"
            ),
            height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠ No data found. Please try again.")

st.markdown("---")
st.caption("🏆 Developed for Smart India Hackathon 2025 | AI Weather Forecasting System 🌦 | Powered by Streamlit, Plotly & scikit-learn")
