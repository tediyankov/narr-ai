
## preliminaries ==============================================================

## libraries
import pandas as pd
from openai import OpenAI as openAI
import os
from tqdm import tqdm

## getting API key
api_key = os.environ.get ("OPENAI_API_KEY3")

## loading in data 
MigNar = pd.read_csv (
    "https://raw.githubusercontent.com/tediyankov/Narrative-Extraction/main/Data/MigNar.csv"
)[['body', 'narrative']]

#MigNar = MigNar.sample(2)

## functions =================================================================

## extractor function
def extractor (text, keywords): 

    # initiating model
    client = openAI(
        api_key = os.environ.get("OPENAI_API_KEY3"),
    )

    # prompt
    prompt = f""" 
    I have this text: {text}

    Extract 5 agent-verb-patient tuples relating to the following keywords: {keywords}. 

    Format this as a Python list. This is very important - only output the Python list and nothing else.

    For example: 
    - ["migrants offer cheap labor", "migrants contribute to economy", "migrants are a burden on public finances", "migrants are associated with crime", "migrants enrich culture"]
    - ["migrants are well integrated in UK society", "migrants are necessary for specific sectors of the economy", "migrants threaten national security"]
    - ["migrants cheat systems", "migrants are vulnerable to discrimination", "migrants need public support"]
    """

    # response
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model = 'gpt-4-turbo',
        temperature = 0.5,
        max_tokens = 50
    )

    # output
    output = response.choices[0].message.content
    return output

## keywords
keywords = ['migrants', 'migrant', 'migration', 'refugees', 'Brexit', 'brexit', 'refugees', 'immigrants', 'immigration', 'immigrant']

## applying extractor
tqdm.pandas (desc = "Extracting tuples")
MigNar['tuples'] = MigNar['body'].progress_apply (extractor, args=(keywords,))

## saving the output to CSV in directory
MigNar.to_csv ('MigNar_micro.csv', index = False)

