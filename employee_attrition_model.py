
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv('employe_attrition.csv')

df.ffill(inplace=True)  


categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']


label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


df['Attrition'] = label_encoder.fit_transform(df['Attrition'])


df = df.apply(pd.to_numeric, errors='coerce')


scaler = StandardScaler()
df[['Age', 'MonthlyIncome', 'DistanceFromHome']] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'DistanceFromHome']])


X = df.drop(columns=['Attrition'])  
y = df['Attrition']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')


joblib.dump(model, 'employee_attrition_model.pkl')


