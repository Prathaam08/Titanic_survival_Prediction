# Titanic Survival Prediction

This project is a machine learning-based prediction system to determine whether a passenger survived the Titanic disaster or not. It uses a Random Forest Classifier to predict survival based on features like passenger class, gender, age, and others.

## Table of Contents

- [Project Overview](#project-overview)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Selection](#model-selection)
- [Performance Analysis](#performance-analysis)
- [Steps to Run the Project](#steps-to-run-the-project)

## Project Overview

This project utilizes the Titanic dataset from Kaggle and builds a machine learning pipeline to predict survival outcomes. The project consists of two parts:
1. **Data Preprocessing**: Clean and transform the dataset to be used by machine learning algorithms.
2. **Prediction System**: Build a Flask web app to interact with users and make predictions based on input data.

## Preprocessing Steps

### 1. **Data Cleaning**
   - **Dropped Irrelevant Columns**: Columns like `PassengerId`, `Name`, `Ticket`, and `Cabin` were dropped as they are not useful for predicting survival.
   - **Handled Missing Values**:
     - `Age`: Missing values were filled with the median value of the column.
     - `Embarked`: Missing values were filled with the most frequent value (mode) of the column.
     - `Fare`: Missing values were filled with the median value of the column.

### 2. **Encoding Categorical Variables**
   - **Label Encoding**: The `Sex` column, which is categorical, was label-encoded into binary values (0 for male and 1 for female).
   - **One-Hot Encoding**: The `Pclass` and `Embarked` columns were one-hot encoded to convert them into numerical features. This is done to ensure that the model can interpret them correctly.

### 3. **Feature Scaling**
   - **Standard Scaling**: The numerical columns `Age`, `SibSp`, `Parch`, and `Fare` were scaled using `StandardScaler` to normalize their values, making the model training more efficient and stable.

### 4. **Feature Selection**
   - The relevant columns were selected and transformed into a format that the model can process. The final dataset was prepared by combining the encoded features and scaling the numerical columns.

## Model Selection

For this project, the **Random Forest Classifier** was chosen due to its ability to handle both numerical and categorical data efficiently and its robustness to overfitting. The model was trained using the preprocessed data with the following hyperparameters:
- `n_estimators=100`: The number of trees in the forest.
- `random_state=42`: A fixed seed for reproducibility.

## Performance Analysis

The model was evaluated using the following metrics:
- **Classification Report**: Provides detailed precision, recall, and F1-score for both classes (Survived, Did Not Survive).
- **Confusion Matrix**: Used to visualize the performance of the classifier by comparing predicted vs. actual labels.

The model's accuracy and precision were deemed acceptable, and further tuning or other algorithms (like Gradient Boosting or XGBoost) can be explored for improved performance.

## Steps to Run the Project

### 1. **Install Dependencies**
   - Clone the repository or download the project files.
   - Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

### 2. **Download Titanic Dataset**
   - Download the Titanic dataset from Kaggle Titanic Competition.
   - Save the dataset as train.csv in the data/ directory.

### 3. **Train the Model**
   - Run the train.py script to preprocess the data, train the Random Forest model, and save the model artifacts:

     ```bash
     python train_model.py
    ```
### 4. **Run the Flask App**
    - After training the model, run the Flask web application using the following command:
    ```bash
    python app.py
  ```



