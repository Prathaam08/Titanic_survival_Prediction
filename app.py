from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load artifacts
model = joblib.load('model_files/model.joblib')
scaler = joblib.load('model_files/scaler.joblib')
le = joblib.load('model_files/label_encoder.joblib')
ohe = joblib.load('model_files/onehot_encoder.joblib')
model_columns = joblib.load('model_files/model_columns.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'Pclass': request.form['pclass'],
            'Sex': request.form['sex'],
            'Age': float(request.form['age']),
            'SibSp': int(request.form['sibsp']),
            'Parch': int(request.form['parch']),
            'Fare': float(request.form['fare']),
            'Embarked': request.form['embarked']
        }

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Encode Sex
        input_df['Sex'] = le.transform(input_df['Sex'])

        # One-hot encode Pclass and Embarked
        categorical_input = input_df[['Pclass', 'Embarked']]
        encoded_input = ohe.transform(categorical_input)
        encoded_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(['Pclass', 'Embarked']))

        # Combine features
        input_df = pd.concat([input_df.drop(['Pclass', 'Embarked'], axis=1), encoded_df], axis=1)

        # Scale numerical features
        numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Ensure correct column order
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = 'Survived' if prediction == 1 else 'Did Not Survive'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)