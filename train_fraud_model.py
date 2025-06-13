import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("data/Fraudulent_E-Commerce_Transaction_Data_2.csv")

# Drop irrelevant columns
df.drop(columns=[
    "Transaction ID", "Customer ID", "Transaction Date", "IP Address",
    "Shipping Address", "Billing Address"
], inplace=True)

# Encode categorical features
for col in ["Payment Method", "Product Category", "Customer Location", "Device Used"]:
    df[col] = df[col].astype("category").cat.codes

# --- Correlation Heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Define features and target
X = df.drop("Is Fraudulent", axis=1)
y = df["Is Fraudulent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance classes with SMOTE
X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Feature Importances ---
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Save model and scaler
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully.")
print("📊 Saved: correlation_heatmap.png, feature_importance.png")
