import pandas as pd

df = pd.read_csv("csv/100Hz_normal_2022_03_11_14_34_38.csv")
df.drop(labels=["sequence"], axis=1)

print(df)
