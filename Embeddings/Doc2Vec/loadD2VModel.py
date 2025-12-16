import sys
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

pathSentences = sys.argv[1] #Recibe path de los documentos (un documento por línea)
pathLabels = sys.argv[2] #Recibe las etiquetas correspondientes a los documentos
pathOut = sys.argv[3] #Recibe la ruta de salida y nombre del archivo que contendrá los vectores y etiqueta

print("Generando vectores Doc2Vec a nivel documento...")

model = Doc2Vec.load("model/doc2vec.dbowModel")

fileSentences = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embeds.obj","wb")
fileOutLabels = open(pathOut+"labels.obj","wb")
#fileOut = open(pathOut+"embeds.txt","w")
#fileOut.write("vector;etiqueta\n")
embeds = []
labels = []
for sentence,label in zip(fileSentences,fileLabels):
	tokens = tokenize(sentence,lowercase=True)	
	vector = model.infer_vector(tokens)
	embeds.append(vector)
	labels.append(label)
#	fileOut.write(str(list(vector))+";"+label)
le = LabelEncoder()
labels = le.fit_transform(labels)

pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileSentences.close()
fileLabels.close()
#fileOut.close()
fileOutEmbeds.close()
fileOutLabels.close()


