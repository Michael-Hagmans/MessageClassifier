import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')
    # documentation on this method found here:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[[0]].iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x[:(len(x) - 2)]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x) - 1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    df.duplicated().sum()
    # drop duplicates
    df_no_duplicates = df.drop_duplicates()
    # check number of duplicates
    df_no_duplicates.duplicated().sum()

    # delete NaN values in categories:
    list_NaN = list(df_no_duplicates['id'][df_no_duplicates['related'].isnull()].index)
    df_no_NaN = df_no_duplicates.drop(list_NaN, axis=0)

    # drop rows where 'related'==2:
    list_2 = list(df_no_NaN['id'][df_no_NaN['related'] == 2].index)
    df_clean = df_no_NaN.drop(list_2, axis=0)

    return df_clean

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql(database_filename, engine, index=False)

engine = create_engine('sqlite:///MessagesData.db')
df.to_sql('MessagesData', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()