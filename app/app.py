from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load pre-trained model (best performing model from your training process)
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define categorical and numerical features for preprocessing
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Preprocessing pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    input_data = {
        'age': request.form['age'],
        'workclass': request.form['workclass'],
        'fnlwgt': request.form['fnlwgt'],
        'education': request.form['education'],
        'education-num': request.form['education-num'],
        'marital-status': request.form['marital-status'],
        'occupation': request.form['occupation'],
        'relationship': request.form['relationship'],
        'race': request.form['race'],
        'sex': request.form['sex'],
        'capital-gain': request.form['capital-gain'],
        'capital-loss': request.form['capital-loss'],
        'hours-per-week': request.form['hours-per-week'],
        'native-country': request.form['native-country']
    }
    
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data
    input_df[num_features] = input_df[num_features].apply(pd.to_numeric, errors='coerce')  # Ensure numerical data
    input_df = preprocessor.transform(input_df)  # Apply preprocessing pipeline

    # Predict salary category using the model
    prediction = model.predict(input_df)

    # Return the result
    if prediction[0] == '>50K':
        return render_template('result.html', prediction_text="The predicted salary is >50K.")
    else:
        return render_template('result.html', prediction_text="The predicted salary is <=50K.")

if __name__ == '__main__':
    app.run(debug=True)
