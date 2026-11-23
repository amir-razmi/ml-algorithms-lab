from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Handles either .xlsx or .csv depending on how the Kaggle archive extracts
df_beans = pd.read_excel("../data/Dry_Bean_Dataset.xlsx")

X = df_beans.drop(columns=["Class"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_beans["Class"])  # 7 classes -> 0..6

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)