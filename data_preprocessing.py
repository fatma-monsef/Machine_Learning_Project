import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv('student-por.csv', sep=';')


df = df.drop_duplicates()


label_encoder = LabelEncoder()
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 
               'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in binary_cols:
    df[col] = label_encoder.fit_transform(df[col])

df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)


for col in ['G3', 'absences']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]



df['Prev_Academic_Avg'] = (df['G1'] + df['G2']) / 2


df['Study_Efficiency'] = df['studytime'] / (df['freetime'] + 1)


df['Total_Alc'] = (df['Dalc'] * 5 + df['Walc'] * 2) / 7


df['Total_Support'] = df['schoolsup'] + df['famsup']


df['Academic_Trend'] = df['G2'] - df['G1']

df['Pressure_Score'] = df['absences'] * (df['failures'] + 1)

cols_to_drop = ['G1', 'G2', 'Dalc', 'Walc']
df.drop(columns=cols_to_drop, inplace=True)


df.reset_index(drop=True, inplace=True)


scaling_cols = ['age', 'absences', 'studytime', 'failures', 
                'Prev_Academic_Avg', 'Study_Efficiency', 'Total_Alc']

scaler = StandardScaler()
df[scaling_cols] = scaler.fit_transform(df[scaling_cols])


df.to_csv('cleaned_student_portuguese.csv', index=False)
print(f"Final Data Shape: {df.shape}")
print("\n--- Top Correlations with G3 ---")
print(df.corr()['G3'].sort_values(ascending=False).head(10))