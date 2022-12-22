import sys
import nltk
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

nltk.download(["wordnet", "punkt", "stopwords"])


def load_data(database_filepath):
    """
    Load data from sqlite database.

    Load data from sqlite database to pandas dataframe.

    Parameters:
    database_filepath (str): Sqlite database filepath

    Returns:
    Tuple: X, Y, category_name

    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("tbl_messages", engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenizing and lemmatizing text

    Tokenizing and lemmatizing text using nltk.

    Parameters:
    text (list): list of text

    Returns:
    list: list of clean tokenized text

    """
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = nltk.WordNetLemmatizer()
    valid_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in nltk.corpus.stopwords.words("english")]
    return valid_tokens


def build_model():
    """
    Implement pipeline for building AI model.

    Implement pipeline for building AI model using sklearn.

    Returns:
    GridSearchCV

    """
    pipeline = Pipeline([
        ("cvt", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(AdaBoostClassifier()))
    ])
    params = {
        'cvt__ngram_range': ((1, 1), (1, 2)),
        'cvt__max_df': (0.75, 1.0),
        'clf__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__learning_rate': [0.0001, 0.001, 0.01, 0.1]
    }

    cv = GridSearchCV(pipeline, param_grid=params, verbose=3, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate AI models.

    Evaluate AI model using sklearn classification report.

    Parameters:
    model (sklearn model object): Messages csv filepath
    X_test (Dataframe): X_test input dataframe
    Y_test (Dataframe): Y_test label dataframe
    category_names (list): list of category name

    Returns:
    DataFrame: classification report

    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        y_true = Y_test[col]
        y_hat = y_pred[:, i]
        print(col)
        print(classification_report(y_true, y_hat))
        print("***" * 10)


def save_model(model, model_filepath):
    """
    Save trained model.

    Save trained model to specific model_filepath.

    Parameters:
    model (sklearn model object): trained model
    model_filepath (str): Model filepath
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
