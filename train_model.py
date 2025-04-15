from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Train a simple Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Make sure the "app" folder exists
os.makedirs("app", exist_ok=True)

# Save the model to app/model.pkl
joblib.dump(model, "app/model.pkl")

print("Model trained and saved to app/model.pkl")
