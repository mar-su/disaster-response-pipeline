import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, label_ranking_loss, label_ranking_average_precision_score
from sklearn.preprocessing import FunctionTransformer
from gensim.sklearn_api import D2VTransformer
from models.commons import vector_tokenize, TopicCentroidDistance, f1_scoring, tokenize
import pickle

def load_data(database_filepath):
    """
    Load the prepared data from a SQLite-file.

    Args:
        database_filepath: Path to the SQLite-file.
    Returns:
        X: DataFrame: Feature vector
        Y: DataFrame: Label vector
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse", engine)
    features = ["message", "genre"]
    drop = ["original", "id"]
    X = df[features]
    Y = df.drop(columns=features + drop)
    return X, Y

def build_model():
    """
    Create a pipeline including feature extraction, classification and grid search.

    Returns:
        A sklearn model-pipeline, wich provides `fit()` and `predict()`.
    """
    pipeline = Pipeline([
        ("ct", ColumnTransformer([
            ("tfidf", Pipeline([
                ("countVectorizer", CountVectorizer(tokenizer=tokenize)),
                ("tfidfTransformer", TfidfTransformer()),
            ]), "message"),
            ("msg2Vec", Pipeline([
                ("tokenizer", FunctionTransformer(func=vector_tokenize, validate=False)),
                ("d2v", D2VTransformer(min_count=1, seed=1)),
            ]), "message"),
            ("centroidDistance", Pipeline([
                ("tokenizer", FunctionTransformer(func=vector_tokenize, validate=False)),
                ("tcd", TopicCentroidDistance()),
            ]), "message"),
            ("genre_onehot", OneHotEncoder(dtype="int"), ["genre"])
        ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = dict(
        clf__estimator__n_estimators=[10, 50],
        ct__centroidDistance__tcd__vector_size=[10, 50, 75],
        ct__msg2Vec__d2v__size=[10, 50, 75]
    )

    return GridSearchCV(pipeline, parameters, scoring=f1_scoring, n_jobs=3)

def evaluate_model(model, X_test, Y_test):
    """
    Print model performance for each class. To show the performance, the
    confusion matrix and the classifcation report is used.
    """
    y_predict = model.predict(X_test)
    for index in range(y_predict.shape[1]):
        print("-"*80)
        print(Y_test.columns[index])
        print(classification_report(Y_test.values[:, index], y_predict[:, index]))
        labels = np.sort(Y_test.iloc[:, index].unique())
        cm = confusion_matrix(Y_test.values[:, index], y_predict[:, index], labels=labels)
        print(pd.DataFrame(cm, columns=labels, index=labels))

def save_model(model, model_filepath):
    """
    Serialize the model and save it as pickle file. Note that you have to import
    `vector_tokenize`, `TopicCentroidDistance`, `f1_scoring` from `commons`. As these
    are used, but not included.

    Args:
        model: sklearn model or pipeline, which should be stored
        model_filepath: destination file path of the stored pickle file
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)

def main():
    """
    Read commandline arguments and print help or load data, build the model,
    evaluate the model and save the model according the given arguments.
    """
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
