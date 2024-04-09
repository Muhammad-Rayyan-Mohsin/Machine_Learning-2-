##421 train, test data > 421
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Q2_Fraud_check.csv')
print(df.dtypes)

##-------------------------------------------Data Preprocessing------------------------------------------------------------------------##

print(pd.get_dummies(df,columns = ['Undergrad', 'Marital.Status', 'Urban']).to_csv('Q2_Fraud_check1.csv', index=False))
df2 = pd.read_csv('Q2_Fraud_check1.csv')

df2['Taxable.Income'] = df2['Taxable.Income'].apply(lambda x: '1' if x > 30000 else '0')

scaler = MinMaxScaler()
WorkExp = df2['Work.Experience'].values.reshape(-1,1)
WorkExp_scaled = scaler.fit_transform(WorkExp)
df2['Work.Experience'] = WorkExp_scaled


CityPop = df2['City.Population'].values.reshape(-1,1)
CityPop_scaled = scaler.fit_transform(CityPop)
df2['City.Population'] = CityPop_scaled

df2['Undergrad_NO'] = df2['Undergrad_NO'].astype(int)
df2['Undergrad_YES'] = df2['Undergrad_YES'].astype(int)
df2['Marital.Status_Divorced'] = df2['Marital.Status_Divorced'].astype(int)
df2['Marital.Status_Single'] = df2['Marital.Status_Single'].astype(int)
df2['Marital.Status_Married'] = df2['Marital.Status_Married'].astype(int)
df2['Urban_NO'] = df2['Urban_NO'].astype(int)
df2['Urban_YES'] = df2['Urban_YES'].astype(int)
df2.to_csv('FINAL.csv', index=False)


##-------------------------------------------Model Training------------------------------------------------------------------------##

X = df2.drop(columns=['Taxable.Income'])  # Features
y = df2['Taxable.Income']  # Target variable

# Split the dataset into training and test sets (421 rows for training, rest for test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=180, random_state=42) # 421 for train, 180 for test

# Initialize the regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

##-------------------------------------------Model Evaluation------------------------------------------------------------------------##

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)

print(report)
