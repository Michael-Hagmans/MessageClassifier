import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    '''
    Create tokens from string input.
    :param text: String that contains a message.
    :return: Output is a list of all tokens contained in the string.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/MessageData222.db')
df = pd.read_sql_table('MessageData222', engine)

# load model
model = joblib.load("../models/classifier.pkl.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Found method for len here:
    # http://www.datasciencemadesimple.com/get-string-length-column-dataframe-python-pandas/
    len_message = df['message'].apply(len)

    # This was helpful to count the words:
    # https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/
    num_words = df['message'].apply(lambda x: len(x.split()))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # tutorial on histograms in v3 found here:
    # https://plot.ly/python/v3/histograms/
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
        },
        {
            'data': [
                Histogram(
                    x=len_message
                )
            ],

            'layout': {
                'title': 'Histogram of Message Length (outliers ignored)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length",
                    'range': [0, 600]
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=num_words
                )
            ],

            'layout': {
                'title': 'Histogram of Word Count (outliers ignored)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Words in a Message",
                    'range': [0, 100]
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()