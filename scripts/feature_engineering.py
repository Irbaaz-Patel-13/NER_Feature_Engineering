import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def compute_sentiment(text):
    """Compute sentiment score using TextBlob and VADER."""
    # Handle empty text or None
    if not text:
        return 0, 0
    
    # TextBlob sentiment (polarity between -1 to 1)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # VADER sentiment (compound score between -1 to 1)
    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)['compound']
    
    return polarity, vader_score

def feature_engineering(input_file, output_file):
    """
    Compute numerical features and combine them with entity counts and sentiment scores.
    
    Parameters:
    - input_file (str): Path to the input CSV file (with NER results).
    - output_file (str): Path to the output CSV file with new feature set.
    """
    # Read the dataset with NER results
    df = pd.read_csv(input_file)

    # Ensure missing values in PERSON, ORG, GPE columns are handled
    df['PERSON'] = df['PERSON'].fillna('')
    df['ORG'] = df['ORG'].fillna('')
    df['GPE'] = df['GPE'].fillna('')
    
    # Initialize additional feature columns
    df['article_length'] = df['title'].apply(lambda x: len(x.split()))  # Article length (number of words)
    
    # Count entities in each category
    df['person_count'] = df['PERSON'].apply(lambda x: len(x.split(',')) if x else 0)  # Count of PERSON entities
    df['org_count'] = df['ORG'].apply(lambda x: len(x.split(',')) if x else 0)  # Count of ORG entities
    df['gpe_count'] = df['GPE'].apply(lambda x: len(x.split(',')) if x else 0)  # Count of GPE entities
    
    # Compute sentiment scores (TextBlob and VADER)
    sentiment_scores = df['title'].apply(lambda x: pd.Series(compute_sentiment(x)))
    df[['textblob_polarity', 'vader_score']] = sentiment_scores
    
    # Assume engagement metrics are present in the dataset as columns 'likes', 'shares', 'comments'
    if 'likes' in df.columns and 'shares' in df.columns and 'comments' in df.columns:
        df['engagement_score'] = df['likes'] + df['shares'] + df['comments']
    else:
        df['engagement_score'] = 0  # If no engagement data, set to 0
    
    # Combine all features in a comprehensive feature set
    df['total_entities'] = df['person_count'] + df['org_count'] + df['gpe_count']  # Total entity count
    
    # Save the new dataset with engineered features
    df.to_csv(output_file, index=False)
    print(f"Feature engineering completed. Output saved to {output_file}")