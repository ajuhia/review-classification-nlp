# Review Classification using Web Scraping and NLP
End-to-end NLP project using real-world scraped review data to build and evaluate classification models, with insights on domain generalization and model performance.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)

## Overview
This project builds an end-to-end Natural Language Processing pipeline to classify customer reviews. It covers data collection through web scraping, text preprocessing, feature engineering using TF-IDF, model training, evaluation, and cross-domain testing. The project also analyzes how well a model trained on one review domain generalizes to other domains.

## Objective
The main goals of this project are to:
- collect and structure review data from a website
- preprocess and normalize raw text
- convert unstructured text into numerical features using TF-IDF
- train and compare multiple machine learning models
- evaluate performance using classification metrics
- study cross-domain model generalization

## Dataset
The review data was scraped from:

`http://mlg.ucd.ie/modules/yalp/`

Three domains were selected:
- Fashion
- Gym
- Hair Salon

Each dataset contains approximately 2000 reviews with the following fields:
- `Review`: raw review text
- `Rating`: label used for classification

## Approach

### 1. Data Collection
Review data was collected using:
- `requests` for fetching webpage content
- `BeautifulSoup` for parsing HTML and extracting review information

For each selected domain, review text and labels were extracted and stored in structured CSV files.

### 2. Data Inspection
Initial analysis was performed to:
- inspect dataset size
- check for null values
- analyze class distribution

Class balance was visualized to understand whether the datasets were significantly imbalanced before modeling.

### 3. Text Preprocessing
A preprocessing pipeline was applied to clean the text data:
- lowercasing
- tokenization
- stopword removal
- POS-aware lemmatization

A processed text column was created and used for downstream feature extraction.

### 4. Train-Test Split
Each dataset was split into:
- 70% training data
- 30% testing data

This ensured consistent evaluation across domains.

### 5. Feature Engineering
Text was transformed into numerical vectors using `TfidfVectorizer`.

Configurations used in later experiments included:
- `max_features=1500`
- `min_df=5`

This helped reduce noise and focus on more informative terms.

### 6. Model Training
The following machine learning models were trained and compared:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

The initial model comparison was performed on the Fashion dataset to identify the strongest baseline model.

### 7. Evaluation
Models were evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report
- Cross-validation
- ROC-AUC

These metrics were used to assess both predictive performance and generalization.

### 8. Cross-Domain Testing
After selecting the best-performing model, cross-domain testing was performed by:
- training on one domain
- testing on the other two domains

This helped evaluate whether learned sentiment patterns transfer across categories.

## Results

### Model Comparison on Fashion Dataset
| Model | Test Accuracy |
|------|---------------:|
| Logistic Regression | ~0.88 |
| Random Forest | ~0.86 |
| Naive Bayes | ~0.79 |

### Domain-wise Performance
- Fashion: ~0.88
- Gym: ~0.90
- Hair Salon: ~0.90

### Cross-Domain Performance
- Models trained on Fashion and Gym generalized relatively well to other domains
- Models trained on Hair Salon performed worse on cross-domain testing

## Key Insights
- Logistic Regression performed best on sparse TF-IDF features
- TF-IDF was effective in representing review text for classification
- Cross-domain performance varied based on vocabulary overlap and label distribution
- Domain-specific wording reduced generalization in some cases

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- BeautifulSoup
- Requests

## Real-World Relevance
This project maps well to real-world use cases such as:
- customer feedback analysis
- sentiment monitoring systems
- review classification pipelines
- support automation and complaint triaging
- chatbot input understanding layers

## Future Improvements
Potential next steps for improving this project include:
- using word embeddings or transformer-based models such as BERT
- applying techniques for handling class imbalance if needed
- experimenting with hyperparameter tuning
- building a reusable prediction pipeline
- deploying the model through a simple API or web app

## Conclusion

This project demonstrates a complete NLP workflow, starting from web scraping and ending with model evaluation and generalization analysis. It highlights the importance of preprocessing, feature engineering, model comparison, and domain-aware evaluation in text classification tasks.
