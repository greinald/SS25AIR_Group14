from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json


def load_pairs_from_triples(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            
            pos = data["positive"]
            neg = data["negative"]
            samples.append(InputExample(texts=[query, pos], label=1.0))  # Positive pair
            samples.append(InputExample(texts=[query, neg], label=0.0))  # Negative pair
    return samples

# Load data
train_samples = load_pairs_from_triples("reranker/triplets.jsonl")
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# Load model (binary classification)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

# Train with default BCEWithLogitsLoss
model.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    warmup_steps=100,
    output_path="reranker/out/model_1"
)
model.save("reranker/out/model_1")
