import pandas as pd

# Create a DataFrame from the sample data
data = pd.read_csv("C:\Ali\DM\Bank.csv")

# Perform data cube aggregation by different dimensions

# Aggregation by 'job' and 'marital'
aggregation1 = data.groupby(['job', 'marital']).agg({
    'balance': 'sum',
    'duration': 'mean'
})
print("Aggregation by 'job' and 'marital':")
print(aggregation1)
print()

# Aggregation by 'education' and 'contact'
aggregation2 = data.groupby(['education', 'contact']).agg({
    'balance': 'mean',
    'campaign': 'sum'
})
print("Aggregation by 'education' and 'contact':")
print(aggregation2)
print()

# Aggregation by 'month' and 'poutcome'
aggregation3 = data.groupby(['month', 'poutcome']).agg({
    'balance': 'max',
    'previous': 'mean'
})
print("Aggregation by 'month' and 'poutcome':")
print(aggregation3)
