# Message Classifier

**Libraries:**

The following libraries are used:
- sys
- pandas
- sqlalchemy
- nltk
- sklearn
- pickle
- json
- plotly
- flask

**Motivation:**

In case of an disaster there is no time for finding most important messages. This app might help as it classifies
incoming messages.

**Files:**
/app
    run.py runs the flask app to be able to view results in a web app.
    /templates
        go.html and master.html are html files that contain structure for web app.

/data
    disaster_categories.csv is a csv file that contains the categories belonging to disaster messages.
    disaster_messages.csv is a csv file that contains messages sent during desasters.
    process_data.py is a python script that preprocesses message data.

/models
    train_classifier.py is a python script that trains a machine learning model and saves it as a pickle file.


**How to run app:**

Run the following command from your command line to process the inputs:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Run the following command from your command line to train a machine learning model:
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Start the app using the following command in your command line:
python run.py
Open the web app in a browser using this link:
https://SPACEID-3001.SPACEDOMAIN
where you need to replace SPACEID and SPACEDOMAIN with the details found when you run
env|grep WORK
in your command line.

**Acknowledgement:**

When I was working on this project I found help here:

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
http://www.datasciencemadesimple.com/get-string-length-column-dataframe-python-pandas/
https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/
https://plot.ly/python/v3/histograms/

