import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the data
data = {
    "Category": [
        "Domain-Specific Applications",
        "Software Development",
        "Learning & Research",
        "Underdeveloped"
    ],
    "OpenAI (%)": [79.63, 70.99, 49.33, 63.64],
    "Multiple LLMs (%)": [5.56, 18.32, 20.00, 9.09],
    "Platform/Framework (%)": [5.56, 7.63, 22.67, 24.24],
    "Others (%)": [9.26, 3.05, 8.00, 3.03]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set Category as index for heatmap
df.set_index("Category", inplace=True)

# Create heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", cbar=True)

# Titles and labels
plt.title("LLM Usage Distribution Across Project Categories", fontsize=14)
plt.xlabel("LLM Type")
plt.ylabel("Project Category")

plt.tight_layout()
plt.show()