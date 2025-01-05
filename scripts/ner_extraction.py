import pandas as pd
from transformers import pipeline
import torch

# Initialize the Hugging Face NER pipeline (consider using a model fine-tuned on your data for more accuracy)
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english", device=0 if torch.cuda.is_available() else -1)

def ner_extraction(input_file, output_file):
    """
    Extract named entities using Hugging Face's pre-trained NER model.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to the output CSV file with NER results.
    """
    # Read the preprocessed dataset
    df = pd.read_csv(input_file)

    # Initialize columns for entity counts and types
    df['PERSON'] = ""
    df['ORG'] = ""
    df['GPE'] = ""
    df['MISC'] = ""

    # Batch processing: Split data into manageable chunks for efficiency
    chunk_size = 100  # Adjust depending on available memory and size of data
    for i in range(0, len(df), chunk_size):
        batch = df.iloc[i:i+chunk_size]
        titles = batch['title'].tolist()
        
        # Apply NER to the batch of titles
        ner_results = ner_model(titles)
        
        # Process the results and categorize them
        for idx, title in enumerate(titles):
            # Initialize the dictionary to store entities for the current article
            entity_dict = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],
                'MISC': []
            }

            # Iterate through each entity detected in the current title
            for entity in ner_results[idx]:
                entity_type = entity['entity']
                if entity_type in ['B-PER', 'I-PER']:
                    entity_dict['PERSON'].append(entity['word'])
                elif entity_type in ['B-ORG', 'I-ORG']:
                    entity_dict['ORG'].append(entity['word'])
                elif entity_type in ['B-LOC', 'I-LOC']:  # Handle GPE as location
                    entity_dict['GPE'].append(entity['word'])
                elif entity_type in ['B-MISC', 'I-MISC']:  # Handle MISC (Miscellaneous)
                    entity_dict['MISC'].append(entity['word'])

            # Assign the categorized entity lists as comma-separated strings in the respective columns
            df.loc[df['title'] == title, 'PERSON'] = ', '.join(entity_dict['PERSON'])
            df.loc[df['title'] == title, 'ORG'] = ', '.join(entity_dict['ORG'])
            df.loc[df['title'] == title, 'GPE'] = ', '.join(entity_dict['GPE'])
            df.loc[df['title'] == title, 'MISC'] = ', '.join(entity_dict['MISC'])

    # Save the resulting dataframe with named entities
    df.to_csv(output_file, index=False)
    print(f"Named entities extracted and saved to {output_file}")

