import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_excel('../data/Sleep_Quality_Dataset_India_700.xlsx')

# Drop unnecessary columns
data = data.drop(['Name', 'State'], axis=1)

# Encode categorical columns
encoder = LabelEncoder()
categorical_cols = ['Gender', 'Caffeine_Intake', 'Exercise', 'Stress', 'Sleep_Quality']

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Split features and target
X = data.drop('Sleep_Quality', axis=1)
y = data['Sleep_Quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, '../model/random_forest_model.pkl')
joblib.dump(scaler, '../model/scaler.pkl')

print("Model and scaler saved successfully.")
