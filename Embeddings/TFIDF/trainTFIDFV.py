from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pathCorpus = "../../adaptData/emotionsCEASE/sentencesCEASE.txt" 

data = open(pathCorpus,"r")

file = open("tfidf.model", "wb")

vectorizer = TfidfVectorizer(stop_words='english',smooth_idf=True)
vectorizer.fit_transform(data)

pickle.dump(vectorizer,file)

file.close()