
import pandas as pd
import networkx as nx
import pickle
import leidenalg
import igraph as ig

# Load the graph
with open("/data/exet5975/thesis/NarrAI/NarrAI_network.pickle", "rb") as f:
    G = pickle.load(f)

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
df = pd.concat([df_pos, df_neg])

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)

# Save the DataFrame as a CSV file
df.to_csv('/data/exet5975/thesis/NarrAI/NarrAI_communities.csv', index=False)