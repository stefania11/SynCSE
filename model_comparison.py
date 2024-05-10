from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import numpy as np

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sjtu-lit/SynCSE-scratch-RoBERTa-base")
model = AutoModel.from_pretrained("sjtu-lit/SynCSE-scratch-RoBERTa-base")

sentences = [
    "green light in laundry",
    "bedroom lights 30%",
    "dim the light in the family room",
    "Lights dimmed by half in the kitchen please",
    "set the thermostat to 70.",
    "increase thermostat by 5°",
    "turn tower fan on.",
    "Close my bedroom shades",
    "Close my bedroom window a little bit.",
    "Unpause dining room vacuum",
    "Change to HBO",
    "set oven to 350",
    "Switch to USB input in living room.",
    "house fan off please",
    "turn tower fan down a little."
]

query_sentence = "turn on the light."

# Function to build index (get embeddings for the sentences)
def build_index(sentences, model, tokenizer):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to search (find most similar sentence)
def search(query_sentence, index_embeddings, model, tokenizer):
    query_input = tokenizer(query_sentence, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).squeeze()
        # Ensure query_embedding is 1D even if it has a single element
        query_embedding = query_embedding.view(-1)  # Flatten if necessary

    similarities = []
    for emb in index_embeddings:
        # Ensure emb is 1D for cosine calculation
        emb = emb.view(-1)
        sim = 1 - cosine(query_embedding.numpy(), emb.numpy())
        similarities.append(sim if not np.isnan(sim) else -1)  # Replace NaN with -1

    # Handle empty similarities list
    if not similarities:
        return None, 0

    most_similar_index = similarities.index(max(similarities))
    return sentences[most_similar_index], max(similarities)

# Build index
index_embeddings = build_index(sentences, model, tokenizer)

# Search for similar sentence
result_sentence, similarity = search(query_sentence, index_embeddings, model, tokenizer)

print(f"Most similar sentence: {result_sentence} (Similarity: {similarity})")
