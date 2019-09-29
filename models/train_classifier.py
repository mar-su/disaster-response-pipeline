import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, label_ranking_loss, label_ranking_average_precision_score
from sklearn.svm import SVC
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse", engine)
    features = ["message", "genre"]
    drop = ["original", "id"]
    X = df[features]
    Y = df.drop(columns=features + drop)
    return X, Y



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


def build_model():
    pipeline = Pipeline([
        ("ct", ColumnTransformer([
            ("msg", Pipeline([
                ("cv", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer()),
            ]), "message"),
            ("genre_onehot", OneHotEncoder(dtype="int"), ["genre"])
        ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = dict(clf__estimator__n_estimators=[5,10,15])

    return GridSearchCV(pipeline, parameters)


def evaluate_model(model, X_test, Y_test):
    y_predict = model.predict(X_test)
    for index in range(y_predict.shape[1]):
        print("-"*80)
        print(Y_test.columns[index])
        print(classification_report(Y_test.values[:, index], y_predict[:, index]))
        labels = np.sort(Y_test.iloc[:, index].unique())
        cm = confusion_matrix(Y_test.values[:, index], y_predict[:, index], labels=labels)
        print(pd.DataFrame(cm, columns=labels))

def save_model(model, model_filepath):
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model... (can take a few minutes)')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
