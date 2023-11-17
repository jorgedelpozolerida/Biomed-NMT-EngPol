from laserembeddings import Laser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def main():
    laser = Laser()

    df = pd.read_csv("medical_corpus_clean.tsv",sep=',')
    df["score"] = None

    for i, row in df.iterrows():
        print(i)
        try:
            embeddings_en = laser.embed_sentences(
                [row["eng"]],
                lang='en'
            )  # lang is only used for tokenization
            embeddings_pl = laser.embed_sentences(
                [row["pol"]],
                lang='pl'
            )  # lang is only used for tokenization

            embeddings_en = embeddings_en.reshape(1, -1)
            embeddings_pl = embeddings_pl.reshape(1, -1)
            similarity = cosine_similarity(embeddings_en, embeddings_pl)[0][0]
            df.at[i,'score'] = similarity
        except Exception:
            continue
    df = df[["id", "score"]]
    df.to_csv("LASER_similarity.tsv", sep=",", index=None)

if __name__ == '__main__':
    main()