import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
df = pd.read_csv("data/complaints.csv")

# Filter relevant products
products = ["Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", "Savings account", "Money transfers"]
df = df[df['Product'].isin(products)]

# Remove rows without narratives
df = df.dropna(subset=['Consumer complaint narrative'])

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned_text'] = df['Consumer complaint narrative'].apply(clean_text)

# Save filtered dataset
df.to_csv("data/filtered_complaints.csv", index=False)

# Distribution visualization
sns.countplot(y='Product', data=df)
plt.title("Complaints by Product")
plt.show()

# Narrative length distribution
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
sns.histplot(df['text_length'], bins=50)
plt.title("Distribution of Complaint Narrative Lengths")
plt.show()
