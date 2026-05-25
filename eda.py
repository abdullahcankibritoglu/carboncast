import pandas as pd
from xgboost import XGBRegressor

def get_data():
    """Fetch energy data from Our World in Data"""
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    return df

# Load data
df = get_data()


df = df[df["iso_code"].notna()]
df = df.sort_values(["country", "year"])


features = ['year', 'gdp', 'other_renewable_consumption', 
            'population', 'fossil_fuel_consumption']
target = 'carbon_intensity_elec'


df.dropna(subset=features, inplace=True)
df.dropna(subset=[target], inplace=True)


train = df[df["year"] <= 2018]
test = df[df["year"] > 2018]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Train model
model = XGBRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"Model Accuracy (R²): {score:.2f}")
print("\nFeature Importance:")

# Display feature importance
fi = pd.Series(model.feature_importances_, index=features)
print(fi.sort_values(ascending=False))

# Data shape info
print(f"\nData Summary:")
print(f"Total rows after cleaning: {len(df)}")
print(f"Countries: {df['country'].nunique()}")
print(f"Years range: {df['year'].min()} - {df['year'].max()}")
