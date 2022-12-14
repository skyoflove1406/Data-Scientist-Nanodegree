import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load 2 csv files using pandas.

    Load and merge dataframe from 2 csv files input.

    Parameters:
    messages_filepath (str): Messages csv filepath
    categories_filepath (str): Categories csv filepath

    Returns:
    DataFrame: Merged 2 dataframe from input on "id" column

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on="id")
    return df


def clean_data(df):
    """
    Clean Dataframe.

    Processing Steps on input Dataframe:
        - Splits the categories column into separate,
        - Clearly named columns
        - Converts values to binary
        - Drops duplicates.

    Parameters:
    df (Dataframe): Merged dataframe from csv files

    Returns:
    DataFrame: Cleaned dataframe

    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = [col[:-2] for col in row.values]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype('int64')

    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset=['message'], inplace=True)
    df.drop(['original', 'child_alone'], inplace=True, axis=1)
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)
    return df


def save_data(df, database_filename):
    """
    Data Storage.

    Create engine and save data to tbl_message.

    Parameters:
    df (Dataframe): Merged dataframe from csv files
    database_filename(str): Sqlite db filename

    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('tbl_messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
