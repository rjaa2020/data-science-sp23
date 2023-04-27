import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('./data/HateSpeech_Binary_Dataset.csv').drop(columns=['Unnamed: 0', 'dataset'])


text = data['text']
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
vectors = pd.DataFrame(vectorizer.transform(text).toarray())


# %%

# analysis:
# to_analyze = 1
# v = vectors[to_analyze][vectors[to_analyze] != 0.0]
print()

# cos_sim = cosine_similarity(vectors[1:3])