
## preliminaries ==============================================================

## libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dotenv
import nltk
import spacy
import itertools
import gensim
import networkx as nx
import os
import pickle
import ast
import argparse

from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.matcher import Matcher
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI as openAI
from tqdm import tqdm
from joblib import Parallel, delayed

## nltk settings
nltk.download('vader_lexicon')

## importing benchmark dataset
MigNar = pd.read_csv('/data/exet5975/thesis/NarrAI/MigNar_micro.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbors', type=int, default=7)
args = parser.parse_args()

## creating network ===========================================================

## tuples object
MigNar['tuples'] = MigNar['tuples'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']') else [])
tuples = [t for sublist in MigNar['tuples'].tolist() for t in sublist]

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
    nearest_neighbors = np.argsort(similarities[i])[-args.n_neighbors:-1]  # Exclude the last index because it's the similarity of tuple1 with itself
    for j in nearest_neighbors:
        tuple2 = tuples[j]
        # Avoid self-loops
        if tuple1 != tuple2:
            # determining the sign of the edge using sentiment
            sentiment1 = G.nodes[tuple1]['sentiment']
            sentiment2 = G.nodes[tuple2]['sentiment']
            edge_sign = 'positive' if (sentiment1 > 0 and sentiment2 > 0) or (sentiment1 < 0 and sentiment2 < 0) else 'negative'
            G.add_edge(tuple1, tuple2, sign=edge_sign)

## saving graph 
with open("/data/exet5975/thesis/NarrAI/NarrAI_network.pickle", "wb") as f:
    pickle.dump(G, f)

## drawing graph
pos = nx.spring_layout(G, k=0.15, iterations=50)
edge_colors = ['green' if G[u][v]['sign'] == 'positive' else 'red' for u, v in G.edges()]
plt.figure(figsize=(12, 12))  # Increase figure size for better readability
nx.draw(G, pos, edge_color = edge_colors, with_labels = False, node_size = 10, font_size = 8)  # Adjust node size and font size
plt.savefig("/data/exet5975/thesis/NarrAI/NarrAI_network.png", dpi=300)  # Increase DPI for better quality