import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

df = pd.read_csv('../data/Telco-Customer-Churn.csv')
df.drop(columns=['customerID'], inplace=True)

df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = df['TotalCharges'].replace('', 0).astype(float)

# Map gender: Male -> 1, Female -> 0
# Map Partner: Yes -> 1, No -> 0
# Map Dependents: Yes -> 1, No -> 0
# Map PhoneService: Yes -> 1, No -> 0
# Map MultipleLines: Yes -> 1, No -> 0, No phone service -> 0      !!!Wheter user have phoneService or not, is determined by the PhoneService column
# Map InternetService with OneHotEncoding
# Map OnlineSecurity: Yes -> 1, No -> 0, No internet service -> 0   !!!Wheter user have internetService or not, is determined by the InternetService column
# Map OnlineBackup: Yes -> 1, No -> 0, No internet service -> 0
# Map DeviceProtection: Yes -> 1, No -> 0, No internet service -> 0
# Map TechSupport: Yes -> 1, No -> 0, No internet service -> 0
# Map StreamingTV: Yes -> 1, No -> 0, No internet service -> 0
# Map StreamingMovies: Yes -> 1, No -> 0, No internet service -> 0

# For Contract i could do feature engineering and create a new column called "Contract_Month" which would be 1, 12, 24
# But since we're using Logistic Regression, i will just use OneHotEncoding for this column

# Map PaperlessBilling: Yes -> 1, No -> 0
# Map PaymentMethod with OneHotEncoding
# Map Churn: Yes -> 1, No -> 0

# Standardize the numerical columns: tenure, MonthlyCharges, TotalCharges


YES_NO_MAP = {
  "Yes": 1,
  "No": 0,
  "No internet service": 0,
  "No phone service": 0,
}

yes_no_transformer = FunctionTransformer(
  lambda X: X.apply(lambda col: col.map(YES_NO_MAP)).astype(int), 
  feature_names_out='one-to-one'
)

binary_cols = [
  "Partner",
  "Dependents",
  "PhoneService",
  "MultipleLines",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "PaperlessBilling",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
]

column_transformer = ColumnTransformer(
  remainder="passthrough",
  transformers=[
    ("onehot_no_drop", OneHotEncoder(), ["InternetService"]),
    ("onehot_drop", OneHotEncoder(drop='first'), ["Contract", "PaymentMethod"]),
    ("gender", OrdinalEncoder(), ["gender"]),
    ("binary_addons", yes_no_transformer, binary_cols),
    ("standardize", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"])
  ]
)

X = df.drop(columns=["Churn"])
y = df["Churn"].map(YES_NO_MAP).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)

column_names = column_transformer.get_feature_names_out()