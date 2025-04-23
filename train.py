import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv('data/train.csv')

def preprocess_data(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    return df

df = preprocess_data(df)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

categorical_cols = ['Pclass', 'Embarked']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cols = ohe.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(categorical_cols))

df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'model_files/model.joblib')
joblib.dump(scaler, 'model_files/scaler.joblib')
joblib.dump(le, 'model_files/label_encoder.joblib')
joblib.dump(ohe, 'model_files/onehot_encoder.joblib')
joblib.dump(list(X.columns), 'model_files/model_columns.joblib')
