import pandas as pd

# Read the data file
data = pd.read_csv("stock_market/dow_jones_index.data")

# Read the names file
with open("stock_market/dow_jones_index.names") as f:
    names = f.read()

print(data.head())
print(names)
