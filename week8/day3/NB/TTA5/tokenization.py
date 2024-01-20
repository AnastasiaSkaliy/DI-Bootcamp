import nltk
import spacy
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

stopwords = stopwords.words('english')


doc = 'Why, sometimes I`ve believed as many as 6 impossible things before breakfast? Some other phrase.'

tokens = word_tokenize(doc)
# sentences = sent_tokenize(doc)
# for sent in sentences:
#     print(sent)

# print(stopwords)

#STEMMING
    
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in tokens]

print(stemmed)


# LEMMATIZATION

text = 'We are learning NLP. I am excited about it.'

doc = nlp(text)

# filtered_doc = [word for word in doc if word not in stopwords.words('english')]

lemmatized = [token.lemma_ for token in doc]

print('lemma: ', lemmatized)





    

