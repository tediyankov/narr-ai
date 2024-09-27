# narr-ai
This repo contains the code behind _NarrAI_ - a novel computational narrative extraction (CNE) algorithm. NarrAI is the first CNE algorithm to use both LLMs and network community detection as enhancements to traditional Bayesian probabilistic topic models. 

The current code loads in a sample dataset of news articles in a CSV format, with the column of interest being `body` containing the unstructured text. The code should work on any text as long as the input is a CSV file, with the text contained in a column named `body`. 

The pipeline is as follows: 
1. `1_NarrAI_micro_narratives.py`: this file takes as input
