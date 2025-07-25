import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/parkinsons.csv")

# Specify feature columns and target
feature_columns = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", 
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", 
    "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "HNR", "RPDE", "DFA", "spread1", "spread2", 
    "D2", "PPE", "Feature22", "Feature23", "Feature24", "Feature25", "Feature26", "Feature27", "Feature28"
]
target_column = "status"

# Extract features and target
X = df[feature_columns]
y = df[target_column]

# Remove rows with NaNs in features or target
data_clean = pd.concat([X, y], axis=1).dropna()
X_clean = data_clean[feature_columns]
y_clean = data_clean[target_column]

# Split into train and test sets (e.g., 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Load saved model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Scale test features
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
