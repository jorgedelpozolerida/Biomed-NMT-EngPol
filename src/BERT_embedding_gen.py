import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

def bert_embed_sentences(sentences, model, tokenizer):
    tokenized_input = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**tokenized_input)
    embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    return embeddings

def main():
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    df = pd.read_csv("KDS_AdvancedNLP_FinalProject/data/medical_corpus_clean.tsv", sep=',')
    df["score"] = None

    start_time = time.time()  # Record the start time

    for i, row in df.iterrows():
        print(f"Processing row {i}")
        try:
            embeddings_en = bert_embed_sentences([row["eng"]], model, tokenizer)
            embeddings_pl = bert_embed_sentences([row["pol"]], model, tokenizer)

            similarity = cosine_similarity(embeddings_en, embeddings_pl)[0][0]
            df.at[i, 'score'] = similarity
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time

    print(f"Total running time: {elapsed_time} seconds")

    print("Saving DataFrame to CSV...")
    df = df[["id", "score"]]
    df.to_csv("KDS_AdvancedNLP_FinalProject/data/BERT_similarity.tsv", sep=",", index=None)
    print("CSV saved successfully.")

if __name__ == '__main__':
    main()