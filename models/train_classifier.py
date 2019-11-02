import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load saved .db file that contains message data and extract information for machine learning.
    :param database_filepath: Indicate the location on your machine where the .db file is saved.
    :return: Output contains three parts: Variables X and Y for machine learning and all category names in
                            a seperate variable category_names.
    '''
    filename_elements = database_filepath.split('/')
    table_name = filename_elements[len(filename_elements) - 1][0:len(filename_elements[len(filename_elements) - 1]) - 3]
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, 'sqlite:///' + database_filepath)

    X = df['message'].values
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1).values
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    '''
    Create tokens from string input.
    :param text: String that contains a message.
    :return: Output is a list of all tokens contained in the string.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    list_tokens = []
    for token in tokens:
        list_tokens.append(lemmatizer.lemmatize(str(token)).lower().strip())
    return list_tokens


def build_model():
    '''
    Create a machine learning model with a pipeline that processes data and creates a prediction.
    :return: Output is a model that is not trained yet.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('Tfidf', TfidfTransformer()),
        ('MOC', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1,1), (1,2)),
        'vect__max_df': (0.6, 0.08, 1.0),
        # information on vect parameters found here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        'Tfidf__smooth_idf': (True, False),
        # information on Tfidf parameters found here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

        'MOC__estimator__n_estimators': [50, 100],
        'MOC__estimator__max_depth': [None, 4]
        # information on RF parameters found here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    }



    model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate how your model performs. Functions prints metrics.
    :param model: Machine learning model that you would like to check.
    :param X_test: Input test data.
    :param Y_test: Output test data.
    :param category_names: Names of categories in a list.
    :return: -
    '''
    y_pred = model.predict(X_test)

    for i in range(len(Y_test[0])):
        y_predicted = y_pred[:, i]
        y_true = Y_test[:, i]
        try:
            print(category_names[i])
            print(classification_report(y_true, y_pred=y_predicted, output_dict=True)['1.0'])
        except:
            print(category_names[i])
            print(i)


def save_model(model, model_filepath):
    '''
    Save your trained model in a pickle file.
    :param model: Model to save.
    :param model_filepath: Path where to save the model.
    :return: -
    '''
    # Found a description on how to do this here:
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    filename = model_filepath + '.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    '''
    Run all functions to load data, build and fit model, evaluate model and save trained model.
    :return:
    '''
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