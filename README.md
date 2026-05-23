<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Gemini_AI-6B48FF?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>

<br/><br/>

```
 ██████╗ █████╗ ██████╗ ██████╗  ██████╗ ███╗   ██╗ ██████╗ █████╗ ███████╗████████╗
██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗████╗  ██║██╔════╝██╔══██╗██╔════╝╚══██╔══╝
██║     ███████║██████╔╝██████╔╝██║   ██║██╔██╗ ██║██║     ███████║███████╗   ██║   
██║     ██╔══██║██╔══██╗██╔══██╗██║   ██║██║╚██╗██║██║     ██╔══██║╚════██║   ██║   
╚██████╗██║  ██║██║  ██║██████╔╝╚██████╔╝██║ ╚████║╚██████╗██║  ██║███████║   ██║   
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝  
```

### *Predict. Simulate. Understand. Act.*

**An ML-powered global energy intelligence platform — built with XGBoost, Streamlit & Gemini AI**

</div>

---

## 🌍 What is CarbonCast?

Most energy dashboards show you data. CarbonCast **interprets** it.

> *"Will Turkey hit its 2030 carbon targets? What happens if renewable energy doubles? Which countries are genuinely green — and which ones are just getting lucky?"*

CarbonCast answers these questions through a combination of **machine learning forecasting**, **interactive what-if simulation**, and **AI-generated analyst commentary** — all in a single web application.

Built on 30+ years of data across 80+ countries, this is not a visualization tool. It's a **decision-support system**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **KPI Dashboard** | Carbon intensity, renewable share, and data range — at a glance |
| 📈 **Trend Visualization** | Interactive time-series of carbon intensity over decades |
| 🤖 **ML Forecasting** | XGBoost model predicts 2030 carbon intensity — R² = 0.74 |
| 🎛️ **What-if Simulator** | Adjust renewable/fossil inputs — see 2030 change in real time |
| 🗺️ **Global Heatmap** | Choropleth world map colored by Carbon Intensity |
| 🏆 **Leaderboard** | Top 10 cleanest & dirtiest countries, ranked and scored |
| 🧠 **AI Analyst** | Gemini-powered automatic insight generation per country |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CarbonCast App                        │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Data    │    │  ML Pipeline │    │  AI Layer     │  │
│  │  Layer   │───▶│              │───▶│               │  │
│  │          │    │  XGBoost     │    │  Gemini API   │  │
│  │  OWID    │    │  R² = 0.74   │    │  Auto Insight │  │
│  │  CSV     │    │              │    │               │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│        │                │                    │           │
│        └────────────────┴────────────────────┘           │
│                         │                                │
│              ┌──────────▼──────────┐                     │
│              │   Streamlit UI      │                     │
│              │   KPI | Map | Sim   │                     │
│              └─────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source:** [Our World in Data — Energy Data](https://github.com/owid/energy-data)

| Property | Value |
|---|---|
| Countries | 200+ (79 after cleaning) |
| Time Range | 1990 – 2022 |
| Total Features | 130+ |
| Target Variable | `carbon_intensity_elec` (gCO₂/kWh) |

**Key columns used:**

| Column | Type | Description |
|---|---|---|
| `carbon_intensity_elec` | float | Grams of CO₂ per kWh of electricity — our **target** |
| `gdp` | float | Gross domestic product — proxy for economic development |
| `population` | float | Total population — scales energy demand |
| `fossil_fuel_consumption` | float | TWh of fossil fuel consumed — direct carbon driver |
| `other_renewable_consumption` | float | TWh of renewables — carbon reducer |
| `iso_code` | string | ISO 3166-1 alpha-3 — used for map rendering |

---

## 🧹 Data Cleaning

Raw data contains regions, continents, and income groups — not just countries. Cleaning pipeline:

```python
# Step 1: Remove non-country rows (Africa, Asia, EU27, etc.)
df = df[df['iso_code'].notna()]
# 23,377 rows → 16,800 rows

# Step 2: Sort for time-series integrity
df = df.sort_values(['country', 'year'])

# Step 3: Drop rows with missing values
df = df.dropna(subset=features)
df = df.dropna(subset=['carbon_intensity_elec'])
# 16,800 rows → ~2,100 rows (79 countries × ~27 years avg)
```

---

## 🔬 ML Methodology

### Why XGBoost?

| Model | R² | Notes |
|---|---|---|
| Linear Regression | ~0.45 | Underfits non-linear relationships |
| Random Forest | ~0.68 | Good but slower |
| **XGBoost** | **0.74** | Best balance of speed and accuracy |

### Time-Aware Train/Test Split

```python
# ✅ CORRECT — train on past, test on future
train = df[df['year'] <= 2018]
test  = df[df['year'] >  2018]
```

### Feature Importance

```
fossil_fuel_consumption        31.5% ████████████████
population                     29.9% ███████████████
other_renewable_consumption    23.3% ████████████
gdp                            12.7% ███████
year                            2.6% █
```

---

## 🏆 Sustainability Score

Each country receives a **0–100 score** via min-max normalization:

```python
min_c = df['carbon_intensity_elec'].min()
max_c = df['carbon_intensity_elec'].max()

df['sustainability_score'] = (max_c - df['carbon_intensity_elec']) / (max_c - min_c) * 100
```

| Rank | Country | Score | Carbon Intensity |
|---|---|---|---|
| 🥇 1 | Iceland | 100.0 | 28 gCO₂/kWh |
| 🥈 2 | Norway | 99.8 | 29 gCO₂/kWh |
| 🥉 3 | Switzerland | 99.2 | 36 gCO₂/kWh |
| ... | ... | ... | ... |
| 79 | Turkmenistan | 0.0 | 1,212 gCO₂/kWh |

---

## 🎛️ What-if Simulator

Users can adjust energy inputs and instantly see how 2030 forecast changes:

```python
# Base: last known values for selected country
last_features = country_df.iloc[-1][features].copy()
last_features['year'] = 2030

# Override with user-defined scenario
last_features['other_renewable_consumption'] = renewable_slider_value
last_features['fossil_fuel_consumption'] = fossil_slider_value

# Predict
prediction = float(model.predict(pd.DataFrame([last_features]))[0])
```

---

## 🧠 AI Analyst (Gemini API)

Each country gets an automatically generated 3-sentence energy analysis using Google's Gemini AI.

---

## 📁 Project Structure

```
Carboncast/
│
├── app.py                  # Main Streamlit application
├── eda.py                  # Exploratory analysis & model training
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.10+
- Google Generative AI API key ([Get it here](https://ai.google.dev/))

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/abdullahcankibritoglu/Carboncast.git
cd Carboncast

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Configuration

The app uses Google Generative AI for AI analysis. You'll need to set your API key:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## 🔭 Potential Extensions

- [ ] **Recursive forecasting** — chain predictions year-by-year to 2030
- [ ] **Prophet / ARIMA** — dedicated time-series models per country
- [ ] **Paris Agreement tracker** — "Will this country hit 1.5°C targets?"
- [ ] **Policy event overlay** — mark major energy policy changes on charts
- [ ] **Clustering** — group countries by energy transition profile
- [ ] **Streamlit Cloud deployment** — one-click public access
- [ ] **Multi-language support** — Turkish, German, French UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| ML Model | XGBoost |
| Data Processing | Pandas |
| Visualization | Plotly Express |
| AI Layer | Google Generative AI (Gemini) |
| Dataset | Our World in Data |

---

## 📝 License

This project is open source and available under the MIT License.

---

<div align="center">

**If this project helped you, consider giving it a ⭐**

*Made with 💚 and a lot of carbon-free electricity*

</div>