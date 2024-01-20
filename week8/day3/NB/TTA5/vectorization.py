from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nltk.download('stopwords')


# doc = 'Why, sometimes I`ve believed as many as 6 impossible things before breakfast. I believe that\'s because I am so hungry before breakfast'

# tokens = word_tokenize(doc)

# stemmer = PorterStemmer()
# stemmed = [stemmer.stem(word) for word in tokens]
# filtered_words = [word for word in stemmed if not word.lower() in stopwords.words('english')]


# bow = Counter(filtered_words)
# print(bow)


# docs = ['This is the first document.','This document is the second document.','And this is the third one.', 'This one is the 4th']

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(docs)

# print("Feature names:", vectorizer.get_feature_names_out())
# print("Vectorized representation:\n", X.toarray())

#output:
# Feature names: ['4th' 'and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
# Vectorized representation:
#  [[0 0 1 1 1 0 0 1 0 1] 
#  [0 0 2 0 1 0 1 1 0 1]
#  [0 1 0 0 1 1 0 1 1 1]
#  [1 0 0 0 1 1 0 1 0 1]]

# Notice that the matrix has 4 rows (each one representing one doc in the docs list), the leng of each row is the lenght of the document, and the order of the words is ignored.

#TF-IDF

# documents = [
#     "I love reading books.",
#     "Reading in the morning is refreshing.",
#     "I love morning coffee."
# ]

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(documents)

# print('Voc:', vectorizer.get_feature_names_out())

# print(tfidf_matrix.toarray())


#vectorization exercise in class:
import json
from sklearn.feature_extraction.text import TfidfVectorizer


with open('famous_poems.json', 'r') as f:
    data = json.load(f)

#cleaning stopwords
processed_poems = []
for poem in data:
    tokens = word_tokenize(poem['text'])
    filtered_tokens = [word for word in tokens if not word.lower() in stopwords.words('english') and word.isalpha()]
    processed_poems.append(' '.join(filtered_tokens))

print(processed_poems[:20])

#Applying TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_poems)


print('td-idf matrix', tfidf_matrix.toarray())


feature_names = vectorizer.get_feature_names_out()
for i, row in enumerate(tfidf_matrix.toarray()):
    max_index = row.argmax()
    most_important_word = feature_names[max_index]
    print(f'most important word in poem{i} is {most_important_word}')
