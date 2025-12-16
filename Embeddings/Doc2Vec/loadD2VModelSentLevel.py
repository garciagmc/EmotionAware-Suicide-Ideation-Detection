import nltk
nltk.download('punkt')
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

print("Generando vectores Doc2Vec a nivel de oración...")

model = Doc2Vec.load("model/doc2vec.dbowModel")

fileSentences = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embedsSent.obj","wb")
fileOutLabels = open(pathOut+"labelsSent.obj","wb")
#fileOut = open(pathOut+"embeds.txt","w")
embeds = []
labels = []
for document,label in zip(fileSentences,fileLabels):
	sentences = sent_tokenize(document)
	documentEmbeds = []
	for sent in sentences:
		tokens = tokenize(sent,lowercase=True)	
		vector = model.infer_vector(tokens)
		documentEmbeds.append(vector)
	labels.append(label)
	embeds.append(documentEmbeds)
#	fileOut.write(str(list(documentEmbeds))+";"+label)
le = LabelEncoder()
labels = le.fit_transform(labels)

pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileSentences.close()
fileLabels.close()
fileOutEmbeds.close()
fileOutLabels.close()

