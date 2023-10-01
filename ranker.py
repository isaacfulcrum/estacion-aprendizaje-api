import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib


# LEGACY CODE
# Note: This code is for reference only. It is not used in the app.

# Load your data
df = pd.read_csv('data.csv')  # Replace with your file path

# Encode the target variable to consecutive integers starting from 0
label_encoder = LabelEncoder()
df['ranking'] = label_encoder.fit_transform(df['ranking'])

# Parse 'prev_scores' into a list of floats
def parse_scores(x):
    if isinstance(x, str):
        return [float(i) for i in x.strip('[]').split(',')]
    else:
        return []

df['prev_scores'] = df['prev_scores'].apply(parse_scores)

# Get the maximum number of previous scores
max_scores = 5

# Dynamically create a column for each previous score
for i in range(max_scores):
    column_name = f'prev_score{i+1}'
    df[column_name] = df['prev_scores'].apply(lambda x: x[i] if i < len(x) else None)

# Drop the 'prev_scores' column as we have individual score columns now
df.drop('prev_scores', axis=1, inplace=True)

# Calculate the average of previous scores
df['avg_prev_scores'] = df.iloc[:, 3:].mean(axis=1)
# Calculate the trend as the difference between the current score and the average of previous scores
df['trend'] = df['score'] - df['avg_prev_scores']

# Split data into features (X) and target (y)
X = df.drop('ranking', axis=1)
y = df['ranking']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# print column names
print(X_train.columns)

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Create and train the XGBoost model
params = {
    'objective': 'multi:softmax',
    'num_class': len(label_encoder.classes_),
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'eval_metric': 'mlogloss'
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
joblib.dump(model, 'model.joblib')

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')