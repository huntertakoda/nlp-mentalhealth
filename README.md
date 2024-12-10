# nlp-mentalhealth
Natural Language Processing Sentimental Analysis of Mental Health Data

# NLP Sentiment Analysis Project

This project performs sentiment analysis using both traditional machine learning models (e.g., Logistic Regression) and advanced transformer-based models (DistilBERT). The focus is on analyzing mental health-related text data and classifying it into various sentiment categories.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to analyze text data and classify sentiments into categories such as **positivity**, **depression**, **anxiety**, and **neutral**. The project includes:
- A baseline model using Logistic Regression.
- A fine-tuned DistilBERT model for advanced analysis.
- Adjusted logic for nuanced cases like "neutral" sentiments.

---

## Features

- **Data Preprocessing**: Text cleaning, tokenization, and vectorization.
- **Baseline Model**: Logistic Regression with TF-IDF vectorizer.
- **Advanced Model**: DistilBERT-based transformer for robust sentiment classification.
- **Custom Adjustments**: Post-prediction logic to handle ambiguous cases.
- **Output File**: Saves predictions (and adjusted predictions) to a CSV file.

---

## Dataset

The dataset contains text entries and their respective sentiment categories:
- **Categories**: Positivity, Depression, Anxiety, and Neutral.
