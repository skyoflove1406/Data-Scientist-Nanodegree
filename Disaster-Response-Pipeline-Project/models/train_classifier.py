import sys
import nltk
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
nltk.download(["wordnet", "punkt", "stopwords"])


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("tbl_messages", engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = nltk.WordNetLemmatizer()
    valid_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in nltk.corpus.stopwords.words("english")]
    return valid_tokens


def build_model():
    pipeline = Pipeline([
        ("cvt", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("multi_outputs", MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        y_true = Y_test[col]
        y_hat = y_pred[:, i]
        print(col)
        print(classification_report(y_true, y_hat))
        print("***"*10)


def save_model(model, model_filepath):
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()