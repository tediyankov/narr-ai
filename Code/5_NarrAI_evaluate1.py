
## preliminaries ==============================================================

## libraries
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

## setting device to use GPU
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

## import data
meso_narratives = pd.read_csv(
    '/data/exet5975/thesis/NarrAI/NarrAI_meso_narratives.csv'
)['meso_narrative'].unique().tolist()

with open ('/data/exet5975/thesis/AlgCompare/compas_narratives.pkl', 'rb') as f:
    compas_narratives = pickle.load(f)

## embeddings ==================================================================

## loading in the model 
model_id = "bert-base-uncased"
model = AutoModel.from_pretrained (model_id).to (device)
tokeniser = tokenizer = AutoTokenizer.from_pretrained (model_id)

## function to compute embeddings
def compute_embeddings (texts):
    inputs = tokenizer (texts, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

## ground truth comparison =====================================================

## results storage df
narrative_recovery_results = pd.DataFrame (columns = ['model', 'score'])

## embeddings
compas_embeddings = compute_embeddings (compas_narratives)
narrai_embeddings = compute_embeddings (meso_narratives)

## cosine similarity
similarity = cosine_similarity (compas_embeddings, narrai_embeddings).mean ()

## storing results
narrative_recovery_results = pd.concat (
    [narrative_recovery_results, 
     pd.DataFrame([{'model': 'NarrAI', 'score': similarity}])], 
     ignore_index = True
     )

## saving results
narrative_recovery_results.to_csv ('/data/exet5975/thesis/NarrAI/cne_narrai_results.csv', index = False)



