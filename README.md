# narr-ai
This repo contains the code behind _NarrAI_ - a novel computational narrative extraction (CNE) algorithm. NarrAI is the first CNE algorithm to use both LLMs and network community detection as enhancements to traditional Bayesian probabilistic topic models. 

The current code loads in a sample dataset of news articles in a CSV format, with the column of interest being `body` containing the unstructured text. The code should work on any text as long as the input is a CSV file, with the text contained in a column named `body`. This code also requires a valid OpenAI API key, which should be saved in a .env file in your working directory.

---

The pipeline is as follows: 
1. **`1_NarrAI_micro_narratives.py`**: this code on `pandas` for data manipulation and `openai` for interfacing with the OpenAI API. The input dataset is a CSV file loaded from a GitHub repository (this logic is fully flexible as long as a CSV file is loaded into the environment. It contains a column: `body` (text data). The function `extractor` extracts agent-verb-patient tuples, using OpenAI’s GPT-4-turbo model, related to narrative-theme keywords (e.g., “migrants offer cheap labor”) from the text data in the `body` column. The extracted tuples are stored in a new column called `tuples` in the original dataset. The final data, including the extracted tuples, is saved as a CSV file.
2. **`2_NarrAI_network.py`**: this code reads in the output CSV from Step 1, and processes these tuples into a format suitable for Doc2Vec training, and generates sentence embeddings. Cosine similarity between the embeddings is calculated to understand the relationships between the tuples. A sentiment analysis using nltk’s Vader is applied to each tuple, and these are then added as nodes to a graph, where edges are created based on nearest neighbors and the polarity of sentiments. The graph is saved in a pickle file and visualized with nodes and color-coded edges indicating the sentiment alignment between nodes, finally saving a secondary output as an image.
3. **`3_NarrAI_leiden.py`**: this code takes the graph pickle object from Step 2 and converts it from a `networkx` format into an `igraph` object for community detection using the Leiden algorithm. Edge attributes are extracted, allowing for the separation of positive and negative edges within the graph. Subgraphs are then created based on these positive and negative relationships, and Leiden is applied to each subgraph to detect communities. The results of the community assignments for both subgraphs are stored in separate DataFrames, which are then concatenated, reorganised, and saved as a CSV file for further analysis or visualisation.
4. **`4_NarrAI_narrative_labels.py`**: This code takes the CSV from Step 3 (agent-verb-patient tuples assigned to communities) and applies a custom function to generate community-level summaries. It first loads the data and retrieves an API key for OpenAI’s GPT-4-turbo model. The labeller function prompts the model to summarize micro-narratives into a concise macro-narrative that reflects both sentiment and content. The data is grouped by community, and the function is applied to each community to generate the macro-narrative. Finally, the results, including the macro-narratives, are saved as a CSV file. The unique values in the macro-narratives column is the narrative decomposition of the unstructured input text.

---

This work is being prepared for publication, with the following working title: 
_NarrAI: Leveraging NLP, LLMs and Network Community Detection for Enhanced Narrative Extraction in the UK Migration Debate_

For any questions, contact Tedi Yankov at teodor.yankov@outlook.com.
