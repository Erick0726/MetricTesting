# **Bias Evaluation Metrics Documentation**

This project provides implementations and methodologies for evaluating bias in natural language processing (NLP) models using the following metrics:

## **Embedding-Based**
1. **WEAT (Word Embedding Association Test)**
2. **SEAT (Sentence Embedding Association Test)**
3. **Embedding-Based CEAT**
4. **Categorical Bias Score (CBS)**

## **Probability-Based**
5. **DisCo (Distributional Correspondence Test)**
6. **LBPS Metric**
7. **CrowS-Pairs Score (CPS)**
8. **All Unmasked Likelihood (AUL)**

## **Generation-Text Based**
9. **Social Group Substitution Test**
10. **Co-occurrence Bias Test**
11. **Demographic Representation**
12. **Stereotypical Association**
13. **Score Parity**
14. **Counterfactual Sentiment**
15. **HONEST**
16. **Gender Polarity**

---

## **Table of Contents**
- [Overview](#overview)
- [Embedding-Based](#embedding-based)
  - [WEAT (Word Embedding Association Test)](#weat-word-embedding-association-test)
  - [SEAT (Sentence Embedding Association Test)](#seat-sentence-embedding-association-test)
  - [Embedding-Based CEAT](#embedding-based-ceat)
  - [Categorical Bias Score (CBS)](#categorical-bias-score-cbs)
- [Probability-Based](#probability-based)
  - [DisCo (Distributional Correspondence Test)](#disco-distributional-correspondence-test)
  - [LBPS Metric](#lbps-metric)
  - [CrowS-Pairs Score (CPS)](#crows-pairs-score-cps)
  - [All Unmasked Likelihood (AUL)](#all-unmasked-likelihood-aul)
- [Generation-Text Based](#generation-text-based)
  - [Social Group Substitution Test](#social-group-substitution-test)
  - [Co-occurrence Bias Test](#co-occurrence-bias-test)
  - [Demographic Representation](#demographic-representation)
  - [Stereotypical Association](#stereotypical-association)
  - [Score Parity](#score-parity)
  - [Counterfactual Sentiment](#counterfactual-sentiment)
  - [HONEST](#honest)
  - [Gender Polarity](#gender-polarity)
- [Usage](#usage)
- [Installing Dependencies](#installing-dependencies)
- [Contributing](#contributing)
- [License](#license)


---

## **Overview**

This repository contains tools for bias evaluation in pre-trained language models. These metrics allow researchers and practitioners to detect and quantify biases related to social group representation in natural language processing.

---

## **WEAT: Word Embedding Association Test**

The **WEAT metric** measures bias by taking two sets of target words and two sets of attribute words and measuring the associations between them.

This link has the glove.6B.50d.txt file which is crucial for the WEAT metric: https://drive.google.com/file/d/1L3_k2SwdAyB3YPzlnzDrscjOjcv9jj0E/view?usp=sharing
### **Steps**
1. Prepare target and attribute word sets.
2. Generate word embeddings using a pre-trained language model.
3. Calculate association scores using cosine similarity.
4. Calculate WEAT score with the difference of summation for every association for each target word.

To run the WEAT test for the Word2Vec model, download the file from this link and put the GoogleNews-vectors-negative300.bin file in the same folder as the weat.py file:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g

---

## **SEAT: Sentence Embedding Association Test**

The **SEAT metric** measures biases in sentence embeddings by adapting the Implicit Association Test (IAT). It computes associations between two target sets (e.g., gendered terms) and two attribute sets (e.g., career- and family-related terms).

### **Steps**
1. Prepare target and attribute word sets.
2. Generate sentence embeddings using a pre-trained language model.
3. Calculate association scores using cosine similarity.
4. Conduct statistical tests (e.g., t-tests) to determine bias significance.

---
## **CEAT: Contextual Embedding Association Test**
The  CEAT metric evaluates biases by analyzing the associations between demographic terms and attributes in contextualized embeddings.

### **Steps**
1. Prepare target (demographic) and attribute word sets (e.g., gendered names and career/family terms).
2. Generate word embeddings using a pre-trained language model.
3.Calculate association scores between target and attribute words using cosine similarity.
4. Compute the CEAT score based on the difference in mean association strengths across groups.
5.Conduct statistical tests (e.g., t-tests) to assess the significance of observed differences.

---



## **DisCo: Distributional Correspondence Test**

The **DisCo metric** evaluates biases based on the contextual distribution of terms. It compares how distributions of contextualized embeddings for different groups differ.

### **Steps**
1. Collect context sentences for terms across groups (e.g., male vs. female).
2. Compute contextualized embeddings for each term using a language model.
3. Measure divergence between distributions using metrics like KL divergence or Wasserstein distance.

---
## **LBPS: Likelihood-Based Parity Score**
The **LBPS metric** evaluates biases by analyzing the likelihoods assigned to gender-associated terms in masked language modeling tasks.

## **Steps**

1. Prepare prompts with a [MASK] token representing the word to be predicted.
2. Define male- and female-associated terms (e.g., "man," "woman," "he," "she").
3. Use a pre-trained language model to generate logits for the [MASK] token.
4. Compute probabilities for male- and female-associated terms using the logits.
5. Calculate the bias score as the absolute difference between male and female probabilities.


---

## **CrowS-Pairs Score (CPS)**

The **CPS metric** measures bias through the use of two sentences, one of which contains bias, and another one which contains less bias. These sentences are stored in the CrowS-Pairs dataset, and models are evaluated by determining which sentence is more likely to appear in that specific model.

### **Steps**
1. Have the CrowS-Pairs dataset ready to use.
2. For each model, go through the CrowS-Pairs dataset and get the probability for each sentence.
3. Measure CPS score by comparing the probability of the more biased sentence appearing in the model with the probability of the less biased sentence appearing.

---

## **Social Group Substitution Test**

The **Social Group Substitution Test** identifies bias by substituting terms related to different social groups (e.g., gender, race) in identical contexts and observing model output differences.

### **Steps**
1. Define sentences with social group terms (e.g., *man/woman*).
2. Replace terms systematically across sentences.
3. Measure changes in logits, probabilities, or other outputs from the language model.
4. Analyze differences to detect bias.

---

## **Co-Occurrence Bias Test**

The **Co-Occurrence Bias Test** identifies bias by examining the co-occurrence of gendered terms and occupation-related terms in a corpus.

### **Steps**
1. Parse a corpus to find sentences containing gendered and occupation terms.
2. Extract embeddings for co-occurring terms using a language model.
3. Compute cosine similarity between embeddings.
4. Aggregate results and calculate bias scores as the difference in similarity for male- and female-associated terms.

---
## **Demographic Representation**

The **Demographic Representation metric** evaluates biases by analyzing the representation of demographic groups in text generated by language models.

## **Steps**

1. Prepare prompts designed to elicit responses related to specific roles or activities (e.g., "Describe a typical day for a teacher").
2. Use a pre-trained language model to generate responses for each prompt.
3. Analyze the generated text for occurrences of male- and female-associated keywords (e.g., "he," "she," "man," "woman").
4. Count the frequency of demographic terms in the generated outputs.
5. Evaluate representation disparities by comparing counts across demographic groups.

---

## **Stereotypical Association**

The **Stereotypical Association metric** measures bias by checking if a model is more likely to generate text speicifc to a social group if a stereotypical term is present, e.g. how likely the model is to generate the word "she" if the word "nurse" is used in the prompt.

### **Steps**
1. Set up a list of prompts to generate text from.
2. Generate text using a model.
3. Calculate how many times a word related to the tested social group is generated for each prompt.
4. Calculate the percentage for each social group and calculate the association score using absolute value.

---
## **Score Parity**
The Score Parity metric evaluates biases by comparing sentiment scores across demographic groups in text classification tasks.

### **Steps**

1. Prepare a dataset containing text samples and associated demographic information (e.g., "male," "female").
2. Use a pre-trained sentiment analysis model to predict sentiment scores for each text sample.
3. Group sentiment scores by demographic and calculate the average score for each group.
4. Perform statistical tests (e.g., t-tests) to assess the significance of differences in scores between groups.
5. Analyze and report any disparities in sentiment predictions across demographics.

---

## **HONEST**

The **HONEST metric** measures bias by using the HurtLex lexicon and finding the number of times a hurtful word is generated by the text. The bias is measured with how likely a model is to generate a sentence, phrase, or word in the HurtLex lexicon.

### **Steps**
1. Have the HurtLex lexicon ready to use.
2. Set up a list of prompts related to social groups to generate text from.
3. Generate text using a model.
4. Calculate how many times a word in the HurtLex lexicon is generated for each prompt and find the percentage for each model.

---

## **Usage**

Follow these steps to run the code and calculate the SEAT (Sentence Embedding Association Test) score:

1. **Prepare Your Data**:
   - Place the JSON file (`seat_data.json`) containing target and attribute sentences in the `json/` directory.

   Example format of `seat_data.json`:
   ```json
   {
       "targ1": {
           "examples": ["white female name 1", "white female name 2"]
       },
       "targ2": {
           "examples": ["black female name 1", "black female name 2"]
       },
       "attr1": {
           "examples": ["positive word 1", "positive word 2"]
       },
       "attr2": {
           "examples": ["negative word 1", "negative word 2"]
       }
   }
   
2. Running SEAT:
   ```bash
   python seat.py

## **Installing Dependencies**

Installing Dependencies:
   ```bash
    pip install sentence-transformers numpy scipy
```
## **Contributing**
 Erick Pelaez Puig, Valeria Giraldo, Jeslyn Chacko, Elnaz Toreihi.


