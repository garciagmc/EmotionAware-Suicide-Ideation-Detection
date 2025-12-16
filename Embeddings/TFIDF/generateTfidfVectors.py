from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pickle
import numpy as np
#import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

print("Generando vectores TF-IDF a nivel de oración...")

pathSentences = sys.argv[1] #Recibe path de los documentos (un documento por línea)
pathLabels = sys.argv[2] #Recibe las etiquetas correspondientes a los documentos
pathOut = sys.argv[3] #Recibe la ruta de salida y nombre del archivo que contendrá los vectores y etiqueta


tfidfModel = open("tfidf.model","rb")
vectorizer = pickle.load(tfidfModel)

fileSentences = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embeds.obj","wb")
fileOutLabels = open(pathOut+"labels.obj","wb")
#fileOut = open(pathOut+"embeds.txt","w")
#fileOut.write("vector;etiqueta\n")

input_matrix = vectorizer.transform(fileSentences).todense()

fileSentences.seek(0)

embeds = []
labels = []
ind = 0
for sentence,label in zip(fileSentences,fileLabels):
	#tokens = tokenize(sentence,lowercase=True)	
	vector = input_matrix[ind].tolist()
	ind = ind +1
	embeds.append(vector[0])
	labels.append(label)
#	fileOut.write(str(vector)+";"+label)
le = LabelEncoder()
labels = le.fit_transform(labels)
#texts = le.inverse_transform(labels)
#print(np.unique(texts))

pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileSentences.close()
fileLabels.close()
#fileOut.close()
fileOutEmbeds.close()
fileOutLabels.close()
