import json
from typing_extensions import final
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

with open("data/processed/chunk.json","r",encoding="utf-8") as f:
    chunks =json.load(f)
en_stopwords=stopwords.words("english")
ps = PorterStemmer()

def tokenize(text):
    text=text.lower()
    text=text.replace("u.s.","us")
    text = re.sub(r"[/\-\u2011\u2013]", " ", text)
    text_token=word_tokenize(text.lower())
    text_token_punt = [re.sub(r"[^\w\s]", "", word) for word in text_token]
    text_token_stopword = [word for word in text_token_punt
                           if word and word not in en_stopwords]

    text_token_stem = [ps.stem(word) for word in text_token_stopword]
    return text_token_stem

tokenized_chunks=[tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def bm25_search(query, k=5):
        query_token=tokenize(query)
        scores=bm25.get_scores(query_token)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # for i in top:
        #     print(round(scores[i], 2), chunks[i]["chunk_id"], chunks[i]["text"][:100])
        return  [(chunks[i]["chunk_id"], float(scores[i])) for i in top]

model = SentenceTransformer("all-MiniLM-L6-v2")
text=[chunk["text"] for chunk in chunks]
chunk_vecs =   np.load("data/processed/chunk_embeddings.npy")

def Embed_search(query,k=5):
    query_vec=model.encode(query)
    query_util=util.cos_sim(query_vec,chunk_vecs)[0]  # measures how similar two embedding are, by checking whether they point in the same direction.
    top = sorted(range(len(query_util)), key=lambda i: query_util[i], reverse=True)[:k]
    # for i in top:
    #     print(round(float(query_util[i]), 2), chunks[i]["chunk_id"], chunks[i]["text"][:100])
    return [(chunks[i]["chunk_id"],float(query_util[i]))for i in top]

def hybrid_query(question,k=5):
    result_bm25=bm25_search(question,k=20)
    df_bm25=pd.DataFrame(result_bm25, columns=["chunk_id", "score"])
    df_bm25["rank"]=range(1,len(df_bm25)+1)
    result_embed=Embed_search(question,k=20)
    df_embed=pd.DataFrame(result_embed, columns=["chunk_id", "score"])
    df_embed["rank"]=range(1,len(df_embed)+1)
    final= pd.merge(df_embed,df_bm25, on="chunk_id", how="outer", suffixes=( "_embed","_bm25"))
    final["rrf"] = (1 / (60 + final["rank_bm25"])).fillna(0) + (1 / (60 + final["rank_embed"])).fillna(0) # Reciprocal Rank Fusion
    final = final.sort_values("rrf", ascending=False).reset_index(drop=True)
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    final["date"] = final["chunk_id"].map(lambda cid: chunk_lookup[cid]["date"])
    final["text"] = final["chunk_id"].map(lambda cid: chunk_lookup[cid]["text"])
    return final.head(k)