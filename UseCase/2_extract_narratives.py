
## preliminaries =====================================================================

## libraries
import pandas as pd
from openai import OpenAI as openAI
import os
from tqdm import tqdm
import ast
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import networkx as nx
import numpy as np
import nltk
import igraph as ig
import leidenalg
import pickle

## nltk settings
nltk.download('vader_lexicon')

## api key
api_key = os.environ.get ("OPENAI_API_KEY3")

## loading in dataset
guardian_corpus = pd.read_csv ('/data/exet5975/thesis/NarrAI/use_case/UseCase/new_guardian_corpus.csv')

## NarrAI: Step 1 ====================================================================

## extractor function
def extractor (text, keywords): 

    # initiating model
    client = openAI(
        api_key = api_key,
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
        model = 'gpt-3.5-turbo',
        temperature = 0.5,
        max_tokens = 50
    )

    # output
    output = response.choices[0].message.content
    return output

## keywords
keywords = ['migrants', 'migrant', 'migration', 'refugees', 'Brexit', 'brexit', 'refugees', 'immigrants', 'immigration', 'immigrant']

## applying extractor
df_with_tuples = guardian_corpus.copy()
tqdm.pandas (desc = "Extracting tuples")
df_with_tuples['tuples'] = df_with_tuples['Body'].progress_apply (extractor, args=(keywords,))

## NarrAI: Step 2 ====================================================================

## tuples object
df_with_tuples['tuples'] = df_with_tuples['tuples'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']') else [])
tuples = [t for sublist in df_with_tuples['tuples'].tolist() for t in sublist]

## preparing data for Doc2Vec
documents = [TaggedDocument(doc, [i]) for i, doc in tqdm(enumerate(tuples), total=len(tuples), desc="Preparing data for Doc2Vec")]

## Doc2Vec
print("Training Doc2Vec model...")
model = Doc2Vec (documents, vector_size = 50, min_count = 1, epochs = 50)

## tuple sentence embeddings
#print("Generating embeddings...")
embeddings = [model.infer_vector (tuple.split()) for tuple in tqdm (tuples, desc="Generating embeddings")]

## pairwise similarities between all tuples
if embeddings:
    #print("Calculating similarities...")
    similarities = cosine_similarity (embeddings)
else:
    print("No embeddings were computed.")

## sentiment analysis
sia = SentimentIntensityAnalyzer()

# create empty graph 
G = nx.Graph()

## nodes
#print("Adding nodes...")
for tuple in tqdm(tuples, desc="Adding nodes"):
    G.add_node(tuple, sentiment=sia.polarity_scores(tuple)['compound'], label=tuple)

## edges
#print("Adding edges...")
for i, tuple1 in tqdm(enumerate(tuples), total=len(tuples), desc="Adding edges"):
    # Get the indices of the 5 nearest neighbors of tuple1
    nearest_neighbors = np.argsort(similarities[i])[-18:-1]  # Exclude the last index because it's the similarity of tuple1 with itself
    for j in nearest_neighbors:
        tuple2 = tuples[j]
        # Avoid self-loops
        if tuple1 != tuple2:
            # determining the sign of the edge using sentiment
            sentiment1 = G.nodes[tuple1]['sentiment']
            sentiment2 = G.nodes[tuple2]['sentiment']
            edge_sign = 'positive' if (sentiment1 > 0 and sentiment2 > 0) or (sentiment1 < 0 and sentiment2 < 0) else 'negative'
            G.add_edge(tuple1, tuple2, sign=edge_sign)

# Convert NetworkX graph to iGraph
G_ig = ig.Graph.from_networkx(G)

# Add edge attributes
G_ig.es['sign'] = [data['sign'] for _, _, data in G.edges(data=True)]

# Separate positive and negative edges
edges_pos = [(e.source, e.target) for e in G_ig.es if e['sign'] == 'positive']
edges_neg = [(e.source, e.target) for e in G_ig.es if e['sign'] == 'negative']

# Get unique vertices from positive and negative edges
vertices_pos = list(set([v for edge in edges_pos for v in edge]))
vertices_neg = list(set([v for edge in edges_neg for v in edge]))

# Create positive and negative subgraphs
G_pos = G_ig.subgraph(vertices_pos)
G_neg = G_ig.subgraph(vertices_neg)

# Perform Leiden community detection
part_pos = leidenalg.find_partition(G_pos, leidenalg.ModularityVertexPartition)
part_neg = leidenalg.find_partition(G_neg, leidenalg.ModularityVertexPartition)

# Create a dictionary where the keys are the nodes and the values are the communities
node_to_community_pos = {G_ig.vs[node]['label']: i for i, community in enumerate(part_pos) for node in community}
node_to_community_neg = {G_ig.vs[node]['label']: i for i, community in enumerate(part_neg) for node in community}

# Create a DataFrame from the dictionaries
df_pos = pd.DataFrame.from_dict(node_to_community_pos, orient='index', columns=['Community'])
df_neg = pd.DataFrame.from_dict(node_to_community_neg, orient='index', columns=['Community'])

# Reset the index of the DataFrame
df_pos.reset_index(inplace=True)
df_neg.reset_index(inplace=True)

# Rename the columns
df_pos.columns = ['Label', 'Community']
df_neg.columns = ['Label', 'Community']

# Concatenate the DataFrames
communities = pd.concat([df_pos, df_neg])

## NarrAI: Step 3 ====================================================================

## labeller
def labeller (text): 

    # initiating model
    client = openAI(
        api_key = api_key,
    )

    # prompt
    prompt = f""" 
    This is a list of agent-verb-patient tuples, denoting micro-narratives about migrants / migration: {text} 
    
    Generate a summary agent-verb-patient tuple that denotes the macro-narrative of the micro-narratives, giving a very specific and example-backed answer to the following
    core question: "what do migrants do to UK society?d"
    
    It should not be something general like "migrants shape UK society" or "migration influences social dynamics", but rather a specific summary of the micro-narratives.
    For example: "migrants are essential contributors to the UK economy", "migrants are associated with crime in the UK", "migrants are vulnerable and they need help from the public" 
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
        model = 'gpt-3.5-turbo',
        temperature = 0.5,
        max_tokens = 50
    )

    # output
    output = response.choices[0].message.content
    return output

## implement
communities2 = communities.copy()
grouped = communities2.groupby('Community')['Label'].apply(' '.join)
macro_narratives = grouped.apply (labeller)
communities2['macro_narrative'] = communities2['Community'].map(macro_narratives)

## saving results ====================================================================

guardian_narratives = communities2['macro_narrative'].unique().tolist()

# Save as .txt file
with open('/data/exet5975/thesis/NarrAI/use_case/UseCase/new_guardian_narratives.txt', 'w') as f:
    for item in guardian_narratives:
        f.write("%s\n" % item)

# Save as .pkl file
with open('/data/exet5975/thesis/NarrAI/use_case/UseCase/new_guardian_narratives.pkl', 'wb') as f:
    pickle.dump(guardian_narratives, f)