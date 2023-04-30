import pandas as pd
from scipy.io import arff

print("ARFF data loading...")
data = arff.loadarff('./data/arff/PC5.arff')
train = pd.DataFrame(data[0])
print("Data conversion...")
train.to_csv("PC5.csv", index=False)
