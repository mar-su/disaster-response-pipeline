import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.base import TransformerMixin, BaseEstimator
from sqlalchemy import create_engine
import re
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TopicCentroidDistance(BaseEstimator, TransformerMixin):
    """
    Calculate centroids for each positive instance and calculate distance
    to these centroids.
    """
    def __init__(self, vector_size=10):
        self.vector_size = vector_size
        self.d2v = D2VTransformer(size=vector_size, min_count=1, seed=1)

    def fit(self, X, y):
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

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stopwords_en = stopwords.words("english")
def tokenize(text):
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
    return list(map(tokenize, texts))

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    genre = request.args.get('genre', '')

    input = pd.DataFrame([[query, genre]], columns=["message", "genre"])

    # use model to predict classification for query
    classification_labels = model.predict(input)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        genre=genre,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
