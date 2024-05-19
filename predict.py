import pickle
import pandas as pd


# Load the saved pipeline
with open('transaction_fraud_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Example new data as a list of dictionaries
new_data = [
    {
        'amount': 150.0,
        'source_prefix': 'Whiskers',
        'source_postfix': 2417,
        'dest_prefix': 'Buttercup',
        'dest_postfix': 7761,
        'status': 'success',
        'agent': 'Agent A'
    },
    {
        'amount': 300.0,
        'source_prefix': 'Whiskers',
        'source_postfix': 8554,
        'dest_prefix': 'Noodle',
        'dest_postfix': 6787,
        'status': 'fail',
        'agent': 'Agent B'
    },
    {
        'amount': 1000000,
        'source_prefix': 'Flapjack',
        'source_postfix': 37400,
        'dest_prefix': 'Squishy',
        'dest_postfix': 8775,
        'status': 'success',
        'agent': 'Agent A'
    }
]

# Convert the list of dictionaries to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Predict using the loaded pipeline
predictions = pipeline.predict(new_data_df)
print(predictions)
# Output the predictions along with the input data
new_data_df['prediction'] = predictions

# Print the results
print(new_data_df)