
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load dataset
df = pd.read_csv("water_quality_data_generated.csv")

# Drop rows with any missing values (if any)
df.dropna(inplace=True)

# Features and target
features = ["pH", "Dissolved_Oxygen", "Salinity", "Secchi_Depth", "Water_Depth", "Water_Temp", "Air_Temp"]
X = df[features]
y = df["WQI_Category"]

# Encode target labels (e.g., Excellent → 0, Good → 1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and label encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Train model (RandomForest for stability with tabular data)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "water_quality_model.pkl")

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

new_data = np.array([[7.2, 6.5, 30, 2.5, 5.0, 22.0, 25.0]])  # pH, DO, Salinity, Secchi, Depth, Water_Temp, Air_Temp

# Preprocess and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
category = label_encoder.inverse_transform(prediction)

print("Predicted Water Quality Category:", category[0])


# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()