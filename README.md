# IMDB Sentiment Analysis: Hands-on exercise

Data source: [Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial#description)

## Get started
1. Clone the repo
2. Run setup.sh
3. Activate virtual environment: `source .venv/bin/active`
4. Start the notebook: `jupyter notebook`

## Problem Description
**Given labelled data on IMDB movie reviews, can we train a model that can predict whether a given text has a negative sentiment (0) or positive sentiment (1)?**

## Data
Each movie review is a variable sequence of words and the sentiment of each movie review must be classified. The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given movie review has a positive or negative sentiment.

The data was collected by Stanford researchers and was used in a 2011 paper where a split of 50-50 of the data was used for training and test. An accuracy of 88.89% was achieved.

## What is happening in the notebook
1. Load movie review data (contains review id, review text and sentiment (0 or 1; i.e. negative or positive)
2. Vectorize each review. (i.e. turn texts into sequences (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i))
  - we train the tokenizer with `.fit_on_texts(our_texts)`.
	- we use it and the internal vocabulary that it just learnt to convert our review text into a sequence of indexes using `.texts_to_sequences()`
  - play around with it! after calling .fit_on_texts(), you can explore the `tokenizer` with attributes such as `tokenizer.word_counts` and `tokenizer.word_index`
	- Note: Tokenizer takes in a `num_words` parameter, which is the maximum number of words to work with (if set, tokenization will be restricted to the top `num_words` most common words in the dataset).

3. Preprocess data into a vector that Keras will accept

	- zero pad sequences
    - train_test_split

4. Build and train neural network model
