import numpy as np
from nltk.tokenize import word_tokenize
import re
import pandas as pd

def tokenize(text):
    if isinstance(text, list):
        arrayTokenized = []
        for document in text:
            cleaned_document = clean_text(document)
            arrayTokenized.append(word_tokenize(cleaned_document))
        return arrayTokenized
    else:
        cleaned_text = clean_text(text)
        return word_tokenize(cleaned_text)
    
    
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def build_document_matrix(documents):
    terms = set()
    for document in documents:
        for term in document:
            terms.add(term)

    matrix_rows = []
    for term in terms:
        result = {}
        for document_index, document in enumerate(documents):
            result[f"doc_{document_index}"] = 1 if term in document else 0
        matrix_rows.append(result)
    
    return pd.DataFrame(matrix_rows).to_numpy()

def latent_semantic_indexing(documents, query, k):
    tokenized_documents = tokenize(documents)
    tokenized_query = tokenize(query)
    
    tokenized_documents.append(tokenized_query)
    
    C = build_document_matrix(tokenized_documents)
    U, Sigma, V_T = np.linalg.svd(C[..., :-1].T, full_matrices=False)
    
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    V_T_k = V_T[:k, :]
    
    Sigma_k_inversed = np.linalg.inv(Sigma_k)
    
    docs_reduced = np.matmul(U_k, Sigma_k)
    query_vector = np.linalg.multi_dot([C[..., -1], V_T_k.T, Sigma_k_inversed])
    
    similarities = [cosine_similarity(query_vector, doc_reduced) for doc_reduced in docs_reduced]
    similarity_rounded_float = [round(float(similarity),2) for similarity in similarities]
    
    return similarity_rounded_float
    
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

n = int(input())
documents = []
for _ in range(n):
    documents.append(input())
query = input()
k = int(input())

results = latent_semantic_indexing(documents, query, k)
print(results)
