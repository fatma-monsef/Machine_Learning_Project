import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('student-por.csv', sep=';')


print(df.head())
print(df.info())

print("--------------------------------------------")

print("Advanced Statistics:\n", df.describe())
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nSkewness of Data:\n", df[numeric_cols].skew())

print("--------------------------------------------")


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("--------------------------------------------")

# Outlier Analysis for 'G3' (Final Grade)
Q1 = df['G3'].quantile(0.25)
Q3 = df['G3'].quantile(0.75)

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['G3'] < lower_bound) | 
    (df['G3'] > upper_bound)
]

print("Outliers in G3 (First 5):\n", outliers.head())
plt.figure(figsize=(8,5))
sns.boxplot(x=df['G3'])
plt.title("Outliers in Final Grade (G3)")
plt.show()

print("--------------------------------------------")

# Outlier Analysis for 'studytime'
Q1 = df['studytime'].quantile(0.25)
Q3 = df['studytime'].quantile(0.75)

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['studytime'] < lower_bound) | 
    (df['studytime'] > upper_bound)
]

print("Outliers in Study Time (First 5):\n", outliers.head())
plt.figure(figsize=(8,5))
sns.boxplot(x=df['studytime'])
plt.title("Outliers in Study Time")
plt.show()

print("--------------------------------------------")

# Outlier Analysis for 'absences'
Q1 = df['absences'].quantile(0.25)
Q3 = df['absences'].quantile(0.75)

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['absences'] < lower_bound) | 
    (df['absences'] > upper_bound)
]

print("Outliers in Absences (First 5):\n", outliers.head())
plt.figure(figsize=(8,5))
sns.boxplot(x=df['absences'])
plt.title("Outliers in Absences")
plt.show()

print("--------------------------------------------")

df['Pass'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')

plt.figure(figsize=(7, 5))
sns.countplot(x='Pass', data=df)
plt.title('Class Balance (Pass)')
plt.show()

print("\nClass Distribution (%):\n", df['Pass'].value_counts(normalize=True) * 100)

print("--------------------------------------------")


plt.figure(figsize=(7, 5))
sns.countplot(x='famrel', data=df)
plt.title('Distribution of Family Relationship Quality')
plt.show()

print("\nDistribution of Family Relationship (%):\n", df['famrel'].value_counts(normalize=True) * 100)