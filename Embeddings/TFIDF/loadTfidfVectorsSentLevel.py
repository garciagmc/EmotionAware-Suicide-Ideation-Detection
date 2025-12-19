import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize

pathSentences = sys.argv[1] #Recibe path de los documentos (un documento por línea)
pathLabels = sys.argv[2] #Recibe las etiquetas correspondientes a los documentos
pathOut = sys.argv[3] #Recibe la ruta de salida y nombre del archivo que contendrá los vectores y etiqueta

print("Generando vectores TF-IDF a nivel de oración...")

tfidfModel = open("tfidf.model","rb")
vectorizer = pickle.load(tfidfModel)

fileDocuments = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embedsSent.obj","wb")
fileOutLabels = open(pathOut+"labelsSent.obj","wb")
#fileOut = open(pathOut+"embedsSent.txt","w")

fileDocuments.seek(0)

embeds = []
labels = []
ind = 0

for document,label in zip(fileDocuments,fileLabels):
	sentences = sent_tokenize(document)
	input_matrix = vectorizer.transform(sentences).todense()
	ind = ind +1
	embeds.append(input_matrix)
	labels.append(label)
#	fileOut.write(str(np.array(input_matrix))+";"+label)
le = LabelEncoder()
labels = le.fit_transform(labels)

pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileDocuments.close()
fileLabels.close()
#fileOut.close()
fileOutEmbeds.close()
fileOutLabels.close()






