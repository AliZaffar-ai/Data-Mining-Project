#Data Mining Project

#import libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


data = pd.read_csv("C:\Ali\DM\Bank.csv")

print(data.head(10))
print(data.info())

# Drop unnecessary columns
data.drop(['default', 'housing'], axis=1, inplace=True, errors='ignore')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Step (b): Rule-Based Classification using Decision Tree

X = data.drop("y", axis=1)
y = data["y"]

# One-hot encoding categorical variables
X_encoded = pd.get_dummies(X)

# Replace missing values with mean
imputer = SimpleImputer(strategy='mean')
X_encoded_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(X_encoded_imputed, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step (c): Generating Association Rules and Printing Rules

# Convert all non-boolean columns to boolean (True/False) or binary (0/1)
binary_data = X_encoded_imputed.astype(bool)

frequent_itemsets = apriori(binary_data, min_support=0.1, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Sort the rules by confidence in descending order
rules_sorted = rules.sort_values(by='confidence', ascending=False)

# Print the rules in a tabular format
print("Association Rules:")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence']])

# Step (d): Generating Clusters and Printing Clusters

X_cluster = data[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]

# Replace missing values with mean
X_cluster_imputed = pd.DataFrame(imputer.fit_transform(X_cluster), columns=X_cluster.columns)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_cluster_imputed)
labels = kmeans.labels_
print("Cluster Labels:\n")
for i in range(k):
    print("Cluster", i+1, ":")
    print(X_cluster_imputed[labels == i])
    print()


# Step (e): Visualizing Every Step

# Visualizing Missing Values
sns.heatmap(data.isnull(), cmap='YlGnBu', cbar=True, yticklabels=False)
plt.title('Missing Values')
plt.show()

# Visualizing Decision Tree
from sklearn import tree
plt.figure(figsize=(25, 25))
tree.plot_tree(clf, filled=True, feature_names=X_encoded.columns, class_names=clf.classes_)
plt.show()

# Visualizing Association Rules
plt.figure(figsize=(10, 8))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()

# Visualizing Clusters
plt.scatter(X_cluster_imputed['age'], X_cluster_imputed['balance'], c=labels, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.title('Clusters')
plt.show()

# No of persons got loan
loan_counts = data['y'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=loan_counts.index, y=loan_counts.values)
plt.xlabel('Loan')
plt.ylabel('Count')
plt.title('Number of Persons with Loan')
plt.show()
