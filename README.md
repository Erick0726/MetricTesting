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
13. **Counterfactual Sentiment**
14. **HONEST**
15. **Gender Polarity**

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
  - [Counterfactual Sentiment](#counterfactual-sentiment)
  - [HONEST](#honest)
  - [Gender Polarity](#gender-polarity)
- [Usage](#usage)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)


---

## **Overview**

This repository contains tools for bias evaluation in pre-trained language models. These metrics allow researchers and practitioners to detect and quantify biases related to social group representation in natural language processing.

---

## **SEAT: Sentence Embedding Association Test**

The **SEAT metric** measures biases in sentence embeddings by adapting the Implicit Association Test (IAT). It computes associations between two target sets (e.g., gendered terms) and two attribute sets (e.g., career- and family-related terms).

### **Steps**
1. Prepare target and attribute word sets.
2. Generate sentence embeddings using a pre-trained language model.
3. Calculate association scores using cosine similarity.
4. Conduct statistical tests (e.g., t-tests) to determine bias significance.

---

## **DisCo: Distributional Correspondence Test**

The **DisCo metric** evaluates biases based on the contextual distribution of terms. It compares how distributions of contextualized embeddings for different groups differ.

### **Steps**
1. Collect context sentences for terms across groups (e.g., male vs. female).
2. Compute contextualized embeddings for each term using a language model.
3. Measure divergence between distributions using metrics like KL divergence or Wasserstein distance.

---

## **Social Group Substitution Test**

The **Social Group Substitution Test** identifies bias by substituting terms related to different social groups (e.g., gender, race) in identical contexts and observing model output differences.

### **Steps**
1. Define sentences with social group terms (e.g., *man/woman*).
2. Replace terms systematically across sentences.
3. Measure changes in logits, probabilities, or other outputs from the language model.
4. Analyze differences to detect bias.

---

## **Co-occurrence Bias Test**

The **Co-occurrence Bias Test** identifies bias by examining the co-occurrence of gendered terms and occupation-related terms in a corpus.

### **Steps**
1. Parse a corpus to find sentences containing gendered and occupation terms.
2. Extract embeddings for co-occurring terms using a language model.
3. Compute cosine similarity between embeddings.
4. Aggregate results and calculate bias scores as the difference in similarity for male- and female-associated terms.

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
2. Installing Dependencies:
   ```bash
   pip install sentence-transformers numpy scipy
   
4. Running SEAT:
  ```bash
  python seat.py
