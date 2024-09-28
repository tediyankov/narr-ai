
## Preliminaries ==============================================================

## libraries
import pickle
import pandas as pd
from openai import OpenAI as openAI
import re
import csv 
from tqdm import tqdm
import os

## loading narrative list in from guardian_narratives.pkl
guardian_narratives = [
    'migrants enrich UK society by providing essential labor in various sectors such as healthcare, agriculture, and hospitality',
    'migrants contribute significantly to the UK economy',
    'migrants are attempting dangerous journeys to reach the UK',
    'migrants endure challenges and require support',
    'migrants are seeking economic opportunities in the UK'
]

## loadig in data 
guardian_corpus = pd.read_csv('/data/exet5975/thesis/NarrAI/use_case/UseCase/new_guardian_corpus.csv')

## api key
api_key = os.environ.get ("OPENAI_API_KEY3")

## Scoring each article against each narrative ==================================

## function for producing narrative score
def narrative_score (article, narrative):

    # initiating model
    client = openAI(
        api_key = api_key,
    )

    # prompt
    prompt = f""" 
    This is the body text of an article from the Guardian: {article}.
    This is a narrative: {narrative}.
    If this narrative is present in the article, return a 1. If it is not present, return a 0.
    Return only the number 1 or 0, nothing else!
    """

    # response
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model = 'gpt-3.5-turbo',
        temperature = 0.5,
    )

    # extracting response
    message = response.choices[0].message.content

    # extract only the number from response
    number = re.search (r'\d+', message)
    if number:
        response = int (number.group())
    else:
        response = None
    return response

## CSV key file to identify narratives by their label
with open('narrative_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "Narrative"])

## assigning a number and label to each narrative
for i, narrative in tqdm(enumerate(guardian_narratives, start=1), total=len(guardian_narratives)):
    
    label = f"NAR{i}"
    
    # Write the label and narrative to the CSV file
    with open('narrative_labels.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label, narrative])

    # Create a new column in the DataFrame for each narrative
    guardian_corpus[label] = guardian_corpus['Body'].apply(lambda article: narrative_score(article, narrative))

    # Save the updated DataFrame to a new CSV file
    guardian_corpus.to_csv('expanded_guardian_corpus.csv', index=False)

## saving results ==============================================================
guardian_corpus.to_csv('expanded_guardian_corpus.csv', index = False)

