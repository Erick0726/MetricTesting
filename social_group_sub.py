from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model for sequence classification and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Set the model in evaluation mode
model.eval()

# Sample sentences with different social group terms (gender substitution example)
sentences = {
    "male_version": "John is a doctor. He is very experienced.",
    "female_version": "Jane is a doctor. She is very experienced."
}

# Tokenization and preprocessing
def preprocess_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

# Function to get model predictions (logits)
def get_model_predictions(sentence):
    inputs = preprocess_sentence(sentence)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# Measure bias by comparing logits for different versions of the sentence
logits_male = get_model_predictions(sentences['male_version'])
logits_female = get_model_predictions(sentences['female_version'])

# Display logits for both versions
print("Logits for male version (John):", logits_male)
print("Logits for female version (Jane):", logits_female)

# Compare logits to detect any bias in the output
logits_difference = logits_male - logits_female
print("Difference in logits:", logits_difference)

# If logits_difference is significantly non-zero, it could indicate bias
