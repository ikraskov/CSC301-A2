import pandas as pd

# Load your large dataset
df = pd.read_csv('RecipeNLG_dataset.csv')

# # Randomly sample 20 rows from the dataset
# small_df = df.sample(n=20, random_state=42)  # Use random_state for reproducibility

# # Save the smaller dataset to a new CSV
# small_df.to_csv('small_recipe_dataset.csv', index=False)



fourth_df = df.sample(frac=0.25, random_state=42)  # Sample 25% of the dataset
fourth_df.to_csv('quarter_recipe_dataset.csv', index=False)  # Save the quarter dataset to a new CSV file