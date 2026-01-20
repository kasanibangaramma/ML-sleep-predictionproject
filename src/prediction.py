import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('../model/random_forest_model.pkl')
scaler = joblib.load('../model/scaler.pkl')

# New user data (example)
new_data = pd.DataFrame([{
    'Sleep_Hours': 7,
    'Screen_Time': 3,
    'Caffeine_Intake': 1,  # Yes
    'Exercise': 1,         # Yes
    'Stress': 1,           # Medium
    'Gender': 1            # Male
}])

# Scale input data
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)

# Output result
if prediction[0] == 0:
    print("Predicted Sleep Quality: Poor")
elif prediction[0] == 1:
    print("Predicted Sleep Quality: Average")
else:
    print("Predicted Sleep Quality: Good")
