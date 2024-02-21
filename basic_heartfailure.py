import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Specify the path to the training data
train_data = 'heart_failure_clinical_records_dataset.csv'

# Load the dataset
df = pd.read_csv(train_data)

# Feature selection/engineering
X = df[['age', 'sex', 'ejection_fraction', 'diabetes', 'creatinine_phosphokinase', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'smoking']]
y = df['DEATH_EVENT']

print(X)
print(y)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'ROC-AUC: {roc_auc}')

# Save the model artifacts
joblib.dump(model, 'model.joblib')