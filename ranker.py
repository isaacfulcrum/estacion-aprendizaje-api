from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import pickle

# Load data
data = pd.read_csv('user_data.csv')

# Replace missing values with the median value of that column
data.fillna(data.median(), inplace=True)

# Create a new feature that represents the average score of a user
data['avg_score'] = data['total_score'] / data['num_tests']

# Split data into features (X) and target (y)
X = data.drop('ranking', axis=1)
y = data['ranking']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy: ', accuracy)

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)