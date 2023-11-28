import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from possiblefunctions import is_prime
# Load CSV file into a pandas DataFrame
data = pd.read_csv('C:/Users/HP/Downloads/prism.csv')
data = data.rename(columns={'Unit test case ': 'Unit test case'})
# Replace multiple occurrences of 'Positive' class labels
data['Positive/Negative'] = data['Positive/Negative'].replace({'Positive ': 'Positive'})

# Preprocessing the data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(data['Function'] + ' ' + data['Function description'] + ' ' + data['Unit test case'])
y = data['Positive/Negative']

# Define the parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    # Add other parameters to explore
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# Evaluate the model using cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Initialize the model with best parameters
best_model = grid_search.best_estimator_

# Function to classify new function
def classify_function(input_function, input_description, input_test_case):
    combined_text = input_function + ' ' + input_description + ' ' + input_test_case
    transformed_input = vectorizer.transform([combined_text])
    prediction = best_model.predict(transformed_input)
    return prediction[0]

# Example function inputs to classify
input_function = "is_prime"
input_description = "Check for prime number"
input_test_case ="result = is_prime(6)\nassertEqual(result, False)"


# Use the function to classify
prediction = classify_function(input_function, input_description, input_test_case)
print(f"Predicted class: {prediction}")

# Evaluate model performance
y_pred = best_model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Other evaluation metrics
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)
