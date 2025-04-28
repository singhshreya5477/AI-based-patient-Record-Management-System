import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('data/symptom_disease_dataset.csv')

# Fill missing values
for col in df.columns:
    if 'Symptom' in col:
        df[col] = df[col].fillna('None')

# Get all unique symptoms
symptom_set = set()
for col in df.columns:
    if 'Symptom' in col:
        symptom_set.update(df[col].unique())
symptom_set.discard('None')
all_symptoms = sorted(list(symptom_set))

# Encode symptoms into 0/1 features
encoded_data = []
for idx, row in df.iterrows():
    symptoms_present = row[1:].values
    entry = {symptom: 1 if symptom in symptoms_present else 0 for symptom in all_symptoms}
    entry['Disease'] = row['Disease']
    encoded_data.append(entry)

encoded_df = pd.DataFrame(encoded_data)

# Separate features and target
X = encoded_df.drop('Disease', axis=1)
y = encoded_df['Disease']

# Label encode the target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the new model and other files
joblib.dump(model, 'models/symptom_disease_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'models/symptom_list.pkl')

print("✅ New model retrained and saved successfully!")
