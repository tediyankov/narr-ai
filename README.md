# narr-ai
This repo contains the code behind _NarrAI_ - a novel computational narrative extraction (CNE) algorithm. NarrAI is the first CNE algorithm to use both LLMs and network community detection as enhancements to traditional Bayesian probabilistic topic models. 

The current code loads in a sample dataset of news articles in a CSV format, with the column of interest being `body` containing the unstructured text. The code should work on any text as long as the input is a CSV file, with the text contained in a column named `body`. 

The pipeline is as follows: 
1. `1_NarrAI_micro_narratives.py`: Relies on `pandas` for data manipulation and `openai` for interfacing with the OpenAI API. The input dataset is a CSV file loaded from a GitHub repository (this logic is fully flexible as long as a CSV file is loaded into the environment. It contains a column: `body` (text data). The function `extractor` extracts agent-verb-patient tuples, using OpenAI’s GPT-4-turbo model, related to narrative-theme keywords (e.g., “migrants offer cheap labor”) from the text data in the `body` column. The extracted tuples are stored in a new column called `tuples` in the original dataset. The final data, including the extracted tuples, is saved as a CSV file.
2. 
