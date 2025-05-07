import json
import random
from tqdm import tqdm


def gen_train_triples(negatives_per_positives=3):
    """generates training triples for any model that needs to be trained on triples.
    negatives_per_positives: number of negative samples per positive sample, normal is 3-10
    """
    with open("data/train/training13b_clean.json", "r") as f:
        questions = json.load(f)
        
    with open("data/train/documents.json", "r") as f:
        documents = json.load(f)
        documents = [doc["url"] for doc in documents]
    
    triples = []
    for question in tqdm(questions, desc="Generating triples"):
        positive_docs = question["documents"]
        for positive_doc in positive_docs:
            # get the negative samples
            negative_docs = random.sample(documents, negatives_per_positives)
            # add the triple to the list
            for negative_doc in negative_docs:
                # make sure the negative doc is not in the positive docs
                if negative_doc not in positive_docs:
                    # add the triple to the list
                    triples.append({"query": question["body"], "positive": positive_doc, "negative": negative_doc})
    
    with open("reranker/triples.jsonl", "w") as f:
        for item in triples:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    gen_train_triples(3)