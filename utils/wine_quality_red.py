import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv("../data/winequalityN.csv").dropna()
df_wine["type"] = df_wine["type"].map({"red": 0, "white": 1})
df_wine = df_wine[df_wine["quality"] < 9]  # drop the 5 rare '9' rows

X = df_wine.drop(columns=["quality"])
y = (df_wine["quality"] - 3).astype(int)  # shift ordered scores 3..8 -> 0..5

numeric_cols = [col for col in X.columns if col != "type"]

column_transformer = ColumnTransformer(
    remainder="passthrough",  # keeps binary 'type' column as 0/1
    transformers=[
        ("standardize", StandardScaler(), numeric_cols),
    ],
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)