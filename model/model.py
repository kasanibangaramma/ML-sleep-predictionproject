import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
# 1. Load the dataset
data = pd.read_excel('../data/Sleep_Quality_Dataset_India_700.xlsx')

# 2. Preprocessing (example)
# Encode categorical columns
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Caffeine_Intake'] = label_encoder.fit_transform(data['Caffeine_Intake'])
data['Exercise'] = label_encoder.fit_transform(data['Exercise'])
data['Stress'] = label_encoder.fit_transform(data['Stress'])

# Convert target variable
data['Sleep_Quality'] = label_encoder.fit_transform(data['Sleep_Quality'])

# 3. Split the dataset
X = data.drop('Sleep_Quality', axis=1)
y = data['Sleep_Quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save the model
joblib.dump(model, '../model/random_forest_model.pkl')

print("Model saved successfully!")
