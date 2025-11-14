import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("D:\Java\Titanic-Dataset.csv")  
print("\n===== Dataset Head =====")
print(df.head())
print("\n===== Basic Info =====")
print(df.info())
print("\n===== Missing Values =====")
print(df.isnull().sum())
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\n===== Missing Values After Cleaning =====")
print(df.isnull().sum())
le = LabelEncoder()
for col in categorical_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, drop_first=True)
print("\n===== After Encoding =====")
print(df.head())
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\n===== After Scaling =====")
print(df.head())
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot to Visualize Outliers")
plt.show()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
print("\n===== After Removing Outliers =====")
print(df.describe())
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_dataset.csv'.")
print("\n🎉 Task 1 Completed Successfully!")
