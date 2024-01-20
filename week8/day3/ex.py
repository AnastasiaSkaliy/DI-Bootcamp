import nltk
import ssl
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
# print(stopwords.words('english'))


# def new_func():
#     try:
#         _create_unverified_https_context = ssl._create_unverified_context
#     except AttributeError:
#         pass
#     else:
#         ssl._create_default_https_context = _create_unverified_https_context

#     nltk.download()


# docs = ['This is the first document.','This document is the second document.','And this is the third one.', 'This one is the 4th']

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(docs)

# print("Feature names:", vectorizer.get_feature_names_out())
# print("Vectorized representation:\n", X.toarray())


file_path = '/Users/admin/Desktop/python/di-bootcamp/week8/day3/famous_poems.json'
with open(file_path, 'r') as f:
    data = json.load(f)

#cleaning stopwords
processed_poems = []
for poem in data:
    tokens = word_tokenize(poem['text'])
    filtered_tokens = [word for word in tokens if not word.lower() in stopwords.words('english') and word.isalpha()]
    processed_poems.append(' '.join(filtered_tokens))

#aply TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_poems)

# Step 3: Analyze the Output
# Print the vocabulary (unique words)
print("Vocabulary:", vectorizer.get_feature_names_out())

# Print the TF-IDF values for each document
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
    