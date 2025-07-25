import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset (update path as needed)
df = pd.read_csv("data/parkinsons.csv")

# Remove label column 'status' to get feature columns
feature_columns = list(df.columns)
feature_columns.remove("status")

# Features and labels
X = df[feature_columns]
y = df["status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete — model and scaler saved with 28 features.")
