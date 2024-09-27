
## preliminaries ==============================================================

## libraries
import pickle
import pandas as pd 
import spacy
from spacy.matcher import Matcher

## loading data in
narrai_narratives = pd.read_csv(
    '/data/exet5975/thesis/NarrAI/NarrAI_meso_narratives.csv'
)['meso_narrative'].unique().tolist()

narrai_narratives_kmeans = pd.read_csv(
    '/data/exet5975/thesis/NarrAI/NarrAI_meso_narratives_kmeans.csv'
)['meso_narrative'].unique().tolist()

narrai_narratives_llm = [
    'migrants contribute positively to the economy',
    'migration strains public services',
    'Brexit impacts migration levels',
    'migrants enrich cultural fabric',
    'migration leads to social tensions',
    'migrants fill essential roles in healthcare',
    'migration brings diversity and cultural enrichment',
    'migration challenges social cohesion',
    'migrants drive down wages',
    'migration reshapes cultural fabric',
    'migration puts pressure on healthcare services',
    'migrants receive preferential treatment',
    'migration sparks heated debates',
    'migration impacts the labor market',
    'migrants play crucial role in healthcare',
    'migration leads to increased competition for jobs',
    'migrants are linked to crime',
    'migration enriches society',
    'migration poses challenges to integration',
    'Brexit could decrease migration'
]

narrai_narratives_louvain = pd.read_csv(
    '/data/exet5975/thesis/NarrAI/regLouvain/NarrAI_meso_narratives_regLouvain.csv'
)['meso_narrative'].unique().tolist()

narrai_narratives_unsigned = pd.read_csv(
    '/data/exet5975/thesis/NarrAI/regLeiden/NarrAI_meso_narratives_regLeiden.csv'
)['meso_narrative'].unique().tolist()


## evaluation =================================================================

## defining the pattern
pattern = [{'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'},  # Any noun or proper noun
           {'POS': 'VERB', 'OP': '+'},  # Any verb
           {'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'}]  # Any noun or proper noun

## initialising the Matcher with the shared vocabulary
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
matcher.add("AVP", [pattern])

## narrative lists
#narratives = [narrai_narratives, narrai_narratives_unsigned, narrai_narratives_louvain, narrai_narratives_kmeans, narrai_narratives_llm]
narratives = [narrai_narratives, narrai_narratives_kmeans, narrai_narratives_llm, narrai_narratives_unsigned, narrai_narratives_louvain]
#narrative_names = ["NarrAI (Signed Leiden)", "NarrAI (Regular Leiden)", "NarrAI (Louvain)", "NarrAI (k-means t-SNE)", "NarrAI (LLM)"]
narrative_names = ['NarrAI (Signed Leiden)', 'NarrAI (k-means t-SNE)', 'NarrAI (LLM)', 'NarrAI (Regular Leiden)', 'NarrAI (Regular Louvain)']

## results store
results = []

## iterating 
for i, narrative_list in enumerate (narratives):
    count = 0
    # for each string in the list
    for narrative in narrative_list:
        doc = nlp(narrative)
        matches = matcher(doc)
        # if matches are found, increment the counter
        if matches:
            count += 1
    results.append ([narrative_names[i], count])

## converting the results list to a pandas DataFrame
df = pd.DataFrame (results, columns = ['Model', 'Count'])

df['List Length'] = [len(narrative) for narrative in narratives]
df['Count (norm)'] = df['Count'] / df['List Length']

## saving the results
df.to_csv ('/data/exet5975/thesis/NarrAI/NarrAI_evaluate2_results.csv', index = False)
