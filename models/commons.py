
from sklearn.base import TransformerMixin, BaseEstimator
from gensim.sklearn_api import D2VTransformer
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import make_scorer, f1_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TopicCentroidDistance(BaseEstimator, TransformerMixin):
    """
    Calculate centroids for each positive instance and calculate distance
    to these centroids.
    """
    def __init__(self, vector_size=10):
        """
        Initialize transformer with an internal D2VTransformer.

        Args:
            vector_size: Size of vectors produced by Doc2Vec.
        """
        self.vector_size = vector_size
        self.d2v = D2VTransformer(size=vector_size, min_count=1, seed=1)

    def fit(self, X, y):
        """
        Generate Doc2Vec centroids of instances, where the label is 1.

        Args:
            X: Feature vector
            y: Labels
        """
        X_vec = self.d2v.fit_transform(X)
        self.centroids = np.zeros((y.shape[1], self.vector_size))
        if isinstance(y, pd.DataFrame):
            y = y.values
        for i in range(y.shape[1]):
            if (y[:,i]==1).sum() > 0:
                self.centroids[i,:] = X_vec[y[:,i]==1].mean(axis=0)
        return self

    def transform(self, X):
        X_vec = self.d2v.transform(X)
        distance = np.empty((X_vec.shape[0],self.centroids.shape[0]))
        for i in range(self.centroids.shape[0]):
            distance[:,i] = np.apply_along_axis(lambda x: np.linalg.norm(x - self.centroids[i,:], ord=2), 1, X_vec)
        return distance

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

lemmatizer = WordNetLemmatizer()
stopwords_en = stopwords.words("english")
def tokenize(text):
    """
    Extract tokens from a given message. Exclude stopwords, lemmatize, exclude
    punctation and so on.
    Args:
        text: String: Message or text
    Returns:
        List of extracted tokens
    """
    # tokenize
    words = word_tokenize(text)
    # lower
    words = [word.lower() for word in words]
    # remove stop words
    words = [word for word in words if word not in stopwords_en]
    # punctation (incl. flattening)
    words = [word_part for word in words for word_part in re.split("\W", word) if len(word_part) > 0]
    # identify numbers
    words = [word if re.match("^-?\\d*(\\.\\d+)?$", word) is None else "(number)" for word in words]
    # lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def vector_tokenize(texts):
    """
    Vectorized version of tokenize. Tokenize a whole array of messages.
    Args:
        texts: array-like list of messages
    Return:
        list of token lists
    """
    return list(map(tokenize, texts))

def f1_metric(y_true, y_pred):
    """
    Define a metric to combine all f1-scores for each classification. The f1-score
    respects precision and recall, which is very important in this dataset, as
    it has a clear class imbalance.

    Args:
        y_true: DataFrame or numpy-array with labels
        y_pred: DataFrame or numpy-array with predictions.
    Returns:
        Mean of single weighted f1-scores
    """

    # convert to numpy-arrays, if needed
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    f1_scores = np.empty(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        f1_scores[i] = f1_score(y_true[:, i], y_pred[:, i], average="binary")

    return f1_scores.mean()

f1_scoring = make_scorer(f1_metric)
