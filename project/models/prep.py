import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(RAW_PATH)

# Remove the customer identifier—it is not a predictive feature.
df.drop(columns=["CustomerID"], inplace=True)

# Keep categorical values as strings.
# The training pipeline will one-hot-encode them.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Preserve the purchase class balance across both splits.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
