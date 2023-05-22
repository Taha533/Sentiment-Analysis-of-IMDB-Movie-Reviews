
# Sentiment Analysis of MDB Movie Reviews

This project focuses on sentiment analysis of movie reviews using the IMDb dataset. The dataset consists of 50,000 movie reviews labeled as positive or negative. The main goal of this project is to develop models that can accurately classify the sentiment of movie reviews.

## Dataset

| Label | Number of Samples|
| :-----: | :---: | 
| Positive | 25000| 
| Negative | 25000| 

Dataset Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile

## Data Preprocessing

Before training the models, the dataset undergoes preprocessing steps to prepare it for analysis. The following preprocessing steps are performed:

1. Removal of HTML tags: The dataset is cleaned by removing any HTML tags present in the movie reviews.

2. Removal of stop words: Commonly occurring stop words that do not contribute much to sentiment analysis are removed from the reviews.

3. Removal of single characters: Single characters that are often noise in the text are eliminated.

4. Removal of multiple spaces: Extra spaces between words are reduced to a single space.

## Tokenization and Padding

To prepare the textual data for model training, the sentences are tokenized using a tokenizer. Tokenization involves splitting the text into individual words or tokens. This step helps in creating input sequences for the models.

Furthermore, a padding sequence function provided by Keras is utilized to ensure that all sentences have the same length. In this project, a length of 100 is chosen as the maximum sequence length. Padding sequences is crucial for handling variable-length input and enables efficient batch processing.

## GloVe Word Embeddings

To capture semantic relationships between words, GloVe embeddings are employed. GloVe stands for Global Vectors for Word Representation and provides dense vector representations for words. These embeddings allow for measuring similarity between words based on their vector representations.

In this project, GloVe embeddings are utilized to enhance the models' understanding of word semantics and improve their performance in sentiment analysis.

## Sentiment Analysis Models

Three different models are developed for sentiment analysis of movie reviews:

1. Simple Neural Network: This model architecture consists of a simple feed-forward neural network with fully connected layers. It is trained on the preprocessed movie review data to learn sentiment classification.

2. Convolutional Neural Network (CNN): The CNN model incorporates convolutional1D layers, which are effective in capturing local patterns and features in text data. It is trained to perform sentiment analysis on the movie reviews.

3. Long Short-Term Memory (LSTM): The LSTM model is a type of recurrent neural network (RNN) that is particularly effective in capturing long-term dependencies in sequential data. It is trained on the movie reviews to learn sentiment classification.

## Model Training and Evaluation

Each of the models is trained for 10 epochs using the preprocessed movie review dataset. The models are optimized to learn the sentiment expressed in the reviews, and their performances are evaluated based on accuracy.

Based on the experimental results, it is observed that the LSTM model performs better than the other models for sentiment analysis of movie reviews.

Feel free to explore this project's code and experiment with different models and configurations to enhance sentiment analysis performance on the IMDb movie review dataset.

## Requirements

To run this project, the following dependencies are required:

- Python 3.x
- Numpy
- Pandas
- NLTK
- Keras
- TensorFlow
- Scikit-Learn
- GloVe word embeddings

Please make sure to install the necessary libraries and download the GloVe word embeddings before running the project.

## License
This project is licensed under the [MIT License](https://github.com/Taha533/Sentiment-Analysis-of-IMDB-Movie-Reviews/blob/main/LICENSE).

## Contributions
Contributions to this project are welcome. If you would like.

## References

If you find this project useful, consider referencing the following resources:

- [Stanford NLP Group: GloVe](https://nlp.stanford.edu/projects/glove/)

Note: This project is intended for educational purposes and to showcase the implementation of sentiment analysis using different models.

