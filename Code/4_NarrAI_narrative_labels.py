
## preliminaries ==============================================================

## libraries
import pandas as pd
from openai import OpenAI as openAI
import os
from tqdm import tqdm

## loading in data
MigNar = pd.read_csv (
    "/data/exet5975/thesis/NarrAI/NarrAI_communities.csv"
)

## getting API key
api_key = os.environ.get ("OPENAI_API_KEY3")

## functions ==================================================================

## labeller
def labeller (text): 

    # initiating model
    client = openAI(
        api_key = os.environ.get("OPENAI_API_KEY3"),
    )

    # prompt
    prompt = f""" 
    This is a list of agent-verb-patient tuples, denoting micro-narratives about migrants / migration: {text} 
    Generate a summary agent-verb-patient tuple that denotes the macro-narrative of the micro-narratives. 
    It should not be something general like "migrants shape UK society" or "migration influences social dynamics", but rather a specific summary of the micro-narratives.
    For example: "migrants are essential contributors to the UK economy", "migrants are associated with crime in the UK", "migrants are vulnerable and they need help from the public"
    Also, please make it reflective of positive or negative sentiment about migrants / migration. 
    Return only the macro-narrative tuple, nothing else.
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

## implement ==================================================================

## applying function
MigNar2 = MigNar.copy()
grouped = MigNar2.groupby('Community')['Label'].apply(' '.join)
meso_narratives = grouped.apply (labeller)
MigNar2['meso_narrative'] = MigNar2['Community'].map(meso_narratives)

## saving result as CSV
MigNar2.to_csv ('/data/exet5975/thesis/NarrAI/NarrAI_meso_narratives.csv', index = False)
