import nltk
nltk.download('punkt')
import sys
from bert_serving.client import BertClient
from sklearn.preprocessing import LabelEncoder
import pickle
from nltk.tokenize import sent_tokenize

bc = BertClient()

print("Generando vectores BERT a nivel de oración...")

pathSentences = sys.argv[1] #Recibe path de los documentos (un documento por línea)
pathLabels = sys.argv[2] #Recibe las etiquetas correspondientes a los documentos
pathOut = sys.argv[3] #Recibe la ruta de salida y nombre del archivo que contendrá los vectores y etiqueta


fileDocuments = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embedsSent.obj","wb")
fileOutLabels = open(pathOut+"labelsSent.obj","wb")

embeds = []
labels = []
ind = 0

for document,label in zip(fileDocuments,fileLabels):
	sentences = sent_tokenize(document)
	docSentEmbeds = bc.encode(sentences)
	embeds.append(docSentEmbeds)
	labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileDocuments.close()
fileLabels.close()
fileOutEmbeds.close()
fileOutLabels.close()

