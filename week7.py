# 📊 Analyzing Data with Pandas and Visualizing Results with Matplotlib

# ========================
# Task 1: Load and Explore the Dataset
# ========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: use seaborn style for prettier plots
sns.set(style="whitegrid")

# Example dataset: Iris (can be replaced with any CSV file)
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame

# Inspect first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# No missing values in Iris, but if there were:
# df = df.dropna()   # OR df.fillna(value, inplace=True)


# ========================
# Task 2: Basic Data Analysis
# ========================

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and computing mean
grouped = df.groupby("target").mean()
print("\nAverage values per Species:")
print(grouped)

# Rename species column for readability
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# Identify patterns
print("\nObservations:")
print("- Setosa has smallest sepal & petal sizes on average.")
print("- Virginica has the largest petals overall.")


# ========================
# Task 3: Data Visualization
# ========================

# 1. Line Chart - trends (example: sepal length over index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Line Chart of Sepal Length over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal vs Petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()


# ========================
# Findings / Observations
# ========================

print("\nKey Findings:")
print("1. Setosa species clearly stands out with smaller petal lengths.")
print("2. Virginica species has the largest petals, useful for classification.")
print("3. Histograms suggest sepal width is normally distributed.")
print("4. Scatter plot shows strong correlation between petal length and sepal length.")
