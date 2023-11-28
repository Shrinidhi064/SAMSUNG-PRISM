import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load CSV file into a pandas DataFrame
data = pd.read_csv('C:/Users/HP/Downloads/prism.csv')
data = data.rename(columns={'Unit test case ': 'Unit test case'})

# Creating synthetic data for an empty array scenario
n_samples = 100  # Define the number of synthetic samples
synthetic_data = pd.DataFrame({
    'Function': ['your_function_name'] * n_samples,
    'Function description': ['Your function description'] * n_samples,
    'Unit test case': ['result = your_function_name([])'] * n_samples,
    'Positive/Negative': ['Negative'] * n_samples  # Assuming the expected label for empty arrays is 'Negative'
})

# Append synthetic data to the original dataset
data = pd.concat([data, synthetic_data], ignore_index=True)

# Separate features and target variable
X = data[['Function', 'Function description', 'Unit test case']]
y = data['Positive/Negative']

# Initialize RandomOverSampler to balance classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Create a balanced DataFrame
balanced_data = pd.DataFrame(X_resampled, columns=['Function', 'Function description', 'Unit test case'])
balanced_data['Positive/Negative'] = y_resampled

# Preprocessing the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(balanced_data['Function'] + ' ' + balanced_data['Function description'] + ' ' + balanced_data['Unit test case'])
y = balanced_data['Positive/Negative']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict probabilities instead of labels
predicted_probabilities = clf.predict_proba(X)

# Adjust threshold for classification
threshold = 0.3  # Example threshold value
predicted_labels = ['Negative' if prob[0] > threshold else 'Positive' for prob in predicted_probabilities]

# Calculate accuracy on the training set
accuracy = accuracy_score(y, predicted_labels)
print(f"Accuracy of the model: {accuracy:.2f}")
