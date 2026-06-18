import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading full landmark dataset...")
df = pd.read_csv("asl_landmarks_full.csv")

print(f"✅ Total samples: {len(df)}")
print(f"✅ Labels found: {sorted(df['label'].unique())}")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining on {len(X_train)} samples...")
print("This may take 2-3 minutes with 71k samples...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎉 Model accuracy: {accuracy * 100:.2f}%")
print("\nPer letter accuracy:")
print(classification_report(y_test, y_pred))

with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as asl_model.pkl!")
print("Ready to run the app!")