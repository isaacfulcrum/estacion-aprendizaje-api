import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
# joblib
import joblib

# Load your data
df = pd.read_csv('data.csv')  # Replace with your file path

# Encode the target variable to consecutive integers starting from 0
label_encoder = LabelEncoder()
df['ranking'] = label_encoder.fit_transform(df['ranking'])

joblib.dump(label_encoder, 'bin/label_encoder.joblib')

# Parse 'prev_scores' into a list of floats
def parse_scores(x):
    if isinstance(x, str):
        return [float(i) for i in x.strip('[]').split(',')]
    else:
        return []

df['prev_scores'] = df['prev_scores'].apply(parse_scores)

# Calculate statistics from 'prev_scores' and create new features
df['prev_scores_mean'] = df['prev_scores'].apply(lambda x: np.mean(x) if x else 0.0)
df['prev_scores_max'] = df['prev_scores'].apply(lambda x: np.max(x) if x else 0.0)
df['prev_scores_min'] = df['prev_scores'].apply(lambda x: np.min(x) if x else 0.0)
df['prev_scores_std'] = df['prev_scores'].apply(lambda x: np.std(x) if x else 0.0)

# Drop the original 'prev_scores' column
df.drop('prev_scores', axis=1, inplace=True)
# Split data into features (X) and target (y)
X = df.drop('ranking', axis=1)
y = df['ranking']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a grid of hyperparameter values to search over
param_grid = {
    'max_depth': [3, 4, 5, 6],          # Adjust the range of max_depth
    'learning_rate': [0.01, 0.1, 0.2],  # Adjust the range of learning_rate
    'n_estimators': [50, 100, 200]      # Adjust the range of n_estimators
}

# Create the XGBoost classifier
model = xgb.XGBClassifier()

# Create a grid search object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# save best model
joblib.dump(best_model, 'bin/model.joblib')

print(f'Best Hyperparameters: {best_params}')

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
