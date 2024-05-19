import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Import custom transformers
from custom_transformers import create_features, drop_columns

# Load data
data = pd.read_csv("./data_for_interview.csv")

# Convert the timestamp to a readable date format
data['date'] = pd.to_datetime(data['date'], unit='ms')

# Create features based on the date column
data['weekday'] = data['date'].dt.dayofweek
data['hour'] = data['date'].dt.hour

# Ensure necessary columns are present
data['source'] = data['source_prefix'] + data['source_postfix'].astype(str)
data['destination'] = data['dest_prefix'] + data['dest_postfix'].astype(str)
data['source_destination'] = data['source'] + '_' + data['destination']

# Debug: Check if columns are correctly created
print(data[['source', 'destination', 'source_destination']].head())

# Analyze the top 70 destinations
destination_fraud_rate = data.groupby('destination')['label'].agg(['count', 'sum', 'mean']).reset_index()
top_70_destinations = destination_fraud_rate.sort_values(by=['sum', 'mean'], ascending=False).head(70)['destination'].to_list()
data['top_70_destination'] = data['destination'].apply(lambda x: 1 if x in top_70_destinations else 0)

# Separate features and target
X = data.drop(columns=['label', 'date', 'user', 'weekday', 'hour'])
y = data['label']

# Separate features into numerical and categorical
numerical_features = ['amount']
categorical_features = ['source', 'destination', 'agent', 'status', 'source_destination']

# Define preprocessing steps
preprocessor = Pipeline(steps=[
    ('create_features', FunctionTransformer(create_features, validate=False)),
    ('drop_columns', FunctionTransformer(drop_columns, validate=False)),
    ('preprocess', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]))
])

# Define the full pipeline including preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', TruncatedSVD(n_components=20)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the pipeline using pickle
with open('transaction_fraud_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Pipeline has been saved as 'transaction_fraud_pipeline.pkl'.")