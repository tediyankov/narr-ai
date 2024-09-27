# narr-ai
This repo contains the code behind _NarrAI_ - a novel computational narrative extraction (CNE) algorithm. NarrAI is the first CNE algorithm to use both LLMs and network community detection as enhancements to traditional Bayesian probabilistic topic models. 

The current code loads in a sample dataset of news articles in a CSV format, with the column of interest being `body` containing the unstructured text. The code should work on any text as long as the input is a CSV file, with the text contained in a column named `body`. 

The pipeline is as follows: 
1. **`1_NarrAI_micro_narratives.py`**: Relies on `pandas` for data manipulation and `openai` for interfacing with the OpenAI API. The input dataset is a CSV file loaded from a GitHub repository (this logic is fully flexible as long as a CSV file is loaded into the environment. It contains a column: `body` (text data). The function `extractor` extracts agent-verb-patient tuples, using OpenAI’s GPT-4-turbo model, related to narrative-theme keywords (e.g., “migrants offer cheap labor”) from the text data in the `body` column. The extracted tuples are stored in a new column called `tuples` in the original dataset. The final data, including the extracted tuples, is saved as a CSV file.
2. **`2_NarrAI_network.py`**: Reads in the output CSV from Step 1, and processes these tuples into a format suitable for Doc2Vec training, and generates sentence embeddings. Cosine similarity between the embeddings is calculated to understand the relationships between the tuples. A sentiment analysis using nltk’s Vader is applied to each tuple, and these are then added as nodes to a graph, where edges are created based on nearest neighbors and the polarity of sentiments. The graph is saved in a pickle file and visualized with nodes and color-coded edges indicating the sentiment alignment between nodes, finally saving a secondary output as an image.
3. **`3_NarrAI_leiden.py`**: 
