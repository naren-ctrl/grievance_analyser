from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def is_duplicate(new_text, existing_texts):
    new_emb = model.encode(new_text, convert_to_tensor=True)
    existing_embs = model.encode(existing_texts, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(new_emb, existing_embs)
    return similarity_scores.max().item() > 0.8
