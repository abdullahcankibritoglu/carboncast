import pandas as pd
import streamlit as st
import plotly.express as px
from xgboost import XGBRegressor
import google.generativeai as genai

st.set_page_config(page_title="CarbonCast", page_icon="🌍", layout="wide")

@st.cache_data
def get_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    df = df[df["iso_code"].notna()]
    df = df.sort_values(["country", "year"])
    return df

df = get_data()

features = ['year', 'gdp', 'other_renewable_consumption', 'renewables_consumption', 
            'population', 'fossil_fuel_consumption']
target = 'carbon_intensity_elec'

df.dropna(subset=features, inplace=True)
df.dropna(subset=[target], inplace=True)

@st.cache_resource
def train_model():
    train = df[df["year"] <= 2018]
    test = df[df["year"] > 2018]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]
    
    model = XGBRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

model, score = train_model()

st.title("🚀 CarbonCast: Global Energy & AI Predictor")

country = st.sidebar.selectbox("Select Country", sorted(df["country"].unique()))
country_df = df[df["country"] == country]

st.subheader(f"📊 {country} — Summary")

col1, col2, col3 = st.columns(3)

last = country_df.iloc[-1]

with col1:
    st.metric(
        label="⚡ Latest Carbon Intensity",
        value=f"{last['carbon_intensity_elec']:.1f} gCO₂/kWh")

with col2:
    renew = last.get('renewables_share_energy')
    if pd.notna(renew):
        st.metric(label="🌱 Renewable Share", value=f"{renew:.1f}%")
    else:
        st.metric(label="🌱 Renewable Share", value="N/A")

with col3:
    st.metric(
        label="📅 Data Range",
        value=f"{int(country_df['year'].min())} – {int(country_df['year'].max())}"
    )

# Energy Transition Trend
st.subheader(f"{country} - Energy Transition")
fig = px.line(country_df, x='year', y='carbon_intensity_elec', 
              title="Carbon Intensity Over Time", markers=True)
st.plotly_chart(fig, use_container_width=True)

# 2030 Prediction
last_features = country_df.iloc[-1][features].copy()
last_features["year"] = 2030
prediction = model.predict(pd.DataFrame([last_features]))[0]
st.success(f"📈 2030 Forecast: {prediction:.2f} gCO₂/kWh")
st.caption(f"Model Accuracy: R² = {score:.2f}")

# What-If Simulator
st.subheader("🎛️ What-If Simulator")

renewable_energy_consumption = float(last_features["renewables_consumption"])
fossil_fuel_consumption = float(last_features["fossil_fuel_consumption"])

def slider_bounds(value):
    if value == 0:
        return 0, 1
    else:
        return value * 0.5, value * 2.0

min_r, max_r = slider_bounds(renewable_energy_consumption)
min_f, max_f = slider_bounds(fossil_fuel_consumption)

renewable_energy = st.slider(
    label="Renewable Energy Consumption (TWh)",
    min_value=min_r, max_value=max_r, value=renewable_energy_consumption)

fossil_fuel = st.slider(
    label="Fossil Fuel Consumption (TWh)",
    min_value=min_f, max_value=max_f, value=fossil_fuel_consumption)

last_features['renewables_consumption'] = renewable_energy
last_features['fossil_fuel_consumption'] = fossil_fuel
last_features['year'] = 2030
what_if_prediction = float(model.predict(pd.DataFrame([last_features]))[0])

col1, col2 = st.columns(2)
with col1:
    st.info(f"Renewable: {renewable_energy:.1f} TWh")
with col2:
    st.info(f"Fossil Fuel: {fossil_fuel:.1f} TWh")

st.success(f"🔮 Scenario 2030 Forecast: {what_if_prediction:.2f} gCO₂/kWh")
change = what_if_prediction - prediction
if change < 0:
    st.metric("Change from Base Forecast", f"{change:.2f} gCO₂/kWh", delta=f"{change:.2f}")
else:
    st.metric("Change from Base Forecast", f"{change:.2f} gCO₂/kWh", delta=f"+{change:.2f}")

# Global Heatmap
st.subheader("🌍 Global Carbon Intensity Map")
latest_df = df.groupby("country").last().reset_index()

fig = px.choropleth(
    latest_df,
    locations="iso_code",
    color="carbon_intensity_elec",
    hover_name="country",
    color_continuous_scale="Viridis",
)

fig.update_layout(
    title={
        "text": "Carbon Intensity by Country",
        "x": 0.5,
        "xanchor": "center",
        "font": dict(size=22)
    },
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="LightGray",
        projection_type="natural earth"
    ),
    coloraxis_colorbar=dict(
        title="Carbon Intensity (gCO₂/kWh)",
        ticks="outside"
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.update_traces(marker_line_width=0.5, marker_line_color="white")
st.plotly_chart(fig, use_container_width=True)

# Sustainability Score Calculation
min_c = latest_df["carbon_intensity_elec"].min()
max_c = latest_df["carbon_intensity_elec"].max()

latest_df["sustainability_score"] = (max_c - latest_df["carbon_intensity_elec"]) * 100.0 / (max_c - min_c)

# Leaderboards
top10 = latest_df.sort_values(by="sustainability_score", ascending=False).head(10).reset_index(drop=True)
bottom10 = latest_df.sort_values(by="sustainability_score").head(10).reset_index(drop=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏆 Top 10 Cleanest Countries")
    display_top = top10[["country", "sustainability_score", "carbon_intensity_elec"]].copy()
    display_top.columns = ["Country", "Score", "Carbon Intensity"]
    st.dataframe(display_top, use_container_width=True)

with col2:
    st.subheader("⚠️ Top 10 Most Polluting Countries")
    display_bottom = bottom10[["country", "sustainability_score", "carbon_intensity_elec"]].copy()
    display_bottom.columns = ["Country", "Score", "Carbon Intensity"]
    st.dataframe(display_bottom, use_container_width=True)

# AI Analysis
st.subheader("🤖 AI Analysis")

country_score = latest_df[latest_df['country'] == country]['sustainability_score'].values[0]

prompt = f"""
You are an energy analyst. Analyze the following country's energy data and write a short 3-sentence insight.

Country: {country}
Carbon Intensity: {last['carbon_intensity_elec']:.1f} gCO₂/kWh
Sustainability Score: {country_score:.1f}/100
2030 Prediction: {prediction:.1f} gCO₂/kWh
Fossil Fuel Consumption: {last.get('fossil_fuel_consumption', 'N/A')}
Renewable Consumption: {last.get('other_renewable_consumption', 'N/A')}

Write your analysis in English. Be specific, use the numbers above.
"""

if st.button("🤖 Generate AI Analysis"):
    with st.spinner("Gemini is preparing the analysis..."):
        try:
            # Configure Gemini API
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            if not api_key:
                st.error("❌ GEMINI_API_KEY not found in secrets")
                st.info("💡 Add your API key: Settings → Secrets")
            else:
                genai.configure(api_key=api_key)
                
                # List available models
                available_models = [m.name for m in genai.list_models() 
                                  if 'generateContent' in m.supported_generation_methods]
                
                # Select model
                selected_model = None
                for target in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro']:
                    if target in available_models:
                        selected_model = target
                        break
                
                if not selected_model and available_models:
                    selected_model = available_models[0]

                if selected_model:
                    model_ai = genai.GenerativeModel(selected_model)
                    response = model_ai.generate_content(prompt)
                    
                    if response.text:
                        st.info(f"**Analysis:**\n\n{response.text}")
                    else:
                        st.warning("Analysis generated but returned empty content.")
                else:
                    st.error("No available models found.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("💡 Tip: Add your Gemini API key to secrets (Settings → Secrets)")