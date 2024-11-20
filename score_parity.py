# Step 1: Import necessary libraries
import pandas as pd
from transformers import pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the sentiment analysis model with a specified model name
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Step 3: Prepare your dataset (sample data for demonstration)
# Replace this with actual data loading from IMDb or any other source
data = {
    "review": [
        "I loved this movie! It was fantastic.",
        "This movie was terrible and a waste of time.",
        "What an amazing experience, I would watch it again!",
        "Not my favorite, but it had its moments.",
        "An absolute masterpiece! Highly recommend.",
        "I wouldn't recommend this film to anyone.",
        "The cinematography was beautiful, but the story was lacking.",
        "A delightful film! I enjoyed every minute."
    ],
    "demographic": ["male", "female", "male", "female", "male", "female", "male", "female"]  # Example demographics
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 4: Run predictions and collect scores
df['predictions'] = df['review'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
df['label'] = df['review'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# Step 5: Analyze Score Parity
score_parity_analysis = df.groupby('demographic')['predictions'].mean().reset_index()
print("Average Sentiment Score by Demographic:")
print(score_parity_analysis)

# Step 6: Statistical Testing (T-test)
male_scores = df[df['demographic'] == 'male']['predictions']
female_scores = df[df['demographic'] == 'female']['predictions']

t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
print(f"\nT-Statistic: {t_stat}, P-Value: {p_value}")

# Step 7: Optional - Visualize the results (if desired)
# Set the style of the visualization
sns.set(style="whitegrid")

# Create a boxplot to visualize the distribution of scores by demographic
plt.figure(figsize=(8, 6))
sns.boxplot(x='demographic', y='predictions', data=df)
plt.title('Sentiment Score Distribution by Demographic')
plt.ylabel('Sentiment Score')
plt.xlabel('Demographic')
plt.axhline(y=0.5, color='red', linestyle='--')  # Add a line for neutral sentiment (0.5)
plt.show()
