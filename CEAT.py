import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats

# Load pre-trained BERT model and tokenizer
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()  # Set the model to evaluation mode
print("BERT model and tokenizer loaded successfully.")

# Target words (e.g., male vs. female names)
male_names = ["john", "peter", "michael", "kevin"]
female_names = ["mary", "linda", "susan", "emily"]

# Attribute words (e.g., career vs. family)
career_words = ["engineer", "doctor", "lawyer", "scientist"]
family_words = ["mother", "father", "nurture", "parent"]

# Function to get BERT embeddings for a given word
def get_embedding(word):
    print(f"Getting embedding for: {word}")
    
    # Tokenize the input word
    input_ids = tokenizer.encode(word, return_tensors='pt')
    
    # Get the model output
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Extract the hidden states (use the last layer)
    hidden_states = outputs.hidden_states[-1]
    
    # The word embedding is the first token's embedding (changed from 1 to 0)
    word_embedding = hidden_states[0][0].numpy()
    
    return word_embedding

# Function to calculate cosine similarity between two words
def cosine_similarity(word1, word2):
    embedding1 = get_embedding(word1)
    embedding2 = get_embedding(word2)
    
    return 1 - cosine(embedding1, embedding2)

# Function to calculate the average association strengths between target and attribute words
def calculate_association_strengths(target_words, attribute_words):
    strengths = []
    
    for target in target_words:
        target_strengths = []
        for attribute in attribute_words:
            similarity = cosine_similarity(target, attribute)
            target_strengths.append(similarity)
        strengths.append(np.mean(target_strengths))  # Average similarity for each target
    return np.array(strengths)

# Function to run the experiment and perform statistical tests
def run_experiment():
    print("Running experiment...")

    # Calculate association strengths
    male_career_strength = calculate_association_strengths(male_names, career_words)
    female_career_strength = calculate_association_strengths(female_names, career_words)
    
    male_family_strength = calculate_association_strengths(male_names, family_words)
    female_family_strength = calculate_association_strengths(female_names, family_words)
    
    # Print the mean association strengths
    print("Male-Career Strength:", np.mean(male_career_strength))
    print("Female-Career Strength:", np.mean(female_career_strength))
    
    print("Male-Family Strength:", np.mean(male_family_strength))
    print("Female-Family Strength:", np.mean(female_family_strength))
    
    # Perform t-test to see if the difference in association is significant
    career_ttest = stats.ttest_ind(male_career_strength, female_career_strength)
    family_ttest = stats.ttest_ind(male_family_strength, female_family_strength)
    
    print("\nT-test for Career Words (Male vs Female):", career_ttest)
    print("T-test for Family Words (Male vs Female):", family_ttest)

# Run the experiment
if __name__ == "__main__":
    run_experiment()
