import pandas as pd

df = pd.read_csv('DATA_Gutenberg_Books_Summaries.csv')

def get_data(df = df, word_count_range_start = 500, word_count_range_end = 750, number_of_books = 3):
    df['summary'] = df['summary'].str.replace('__NEWL__', ' ')
    filtered_df = df[(df['word_count'] >= word_count_range_start) & 
                     (df['word_count'] <= word_count_range_end)].copy()
    
    filtered_df['word_count_csv'] = filtered_df['word_count']
    
    filtered_df['real_word_count'] = filtered_df['summary'].str.split().str.len()
    result_df = filtered_df[['title', 'full_text', 'summary', 'word_count_csv', 'real_word_count']].head(number_of_books)
    result_df.to_csv('testing_data.csv', index=False)
    return result_df


def load_testing_data():
    """
    Loads the testing data from testing_data.csv into a pandas DataFrame
    
    Returns:
        pd.DataFrame: DataFrame containing the testing data
    """
    try:
        df = pd.read_csv('testing_data.csv')
        return df
    except Exception as e:
        print(f"Error loading testing data: {e}")
        return None











