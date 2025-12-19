import sys
from bert_serving.client import BertClient
from sklearn.preprocessing import LabelEncoder
import pickle

bc = BertClient()

pathSentences = sys.argv[1] #Recibe path de los documentos (un documento por línea)
pathLabels = sys.argv[2] #Recibe las etiquetas correspondientes a los documentos
pathOut = sys.argv[3] #Recibe la ruta de salida y nombre del archivo que contendrá los vectores y etiqueta

print("Generando vectores BERT...")

fileDocuments = open(pathSentences,"r")
fileLabels = open(pathLabels,"r")
fileOutEmbeds = open(pathOut+"embeds.obj","wb")
fileOutLabels = open(pathOut+"labels.obj","wb")

documents = []
labels = []
ind = 0

for document,label in zip(fileDocuments,fileLabels):
	documents.append(document)
	labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
embeds = bc.encode(documents)
pickle.dump(embeds,fileOutEmbeds)
pickle.dump(labels,fileOutLabels)

fileDocuments.close()
fileLabels.close()
#fileOut.close()
fileOutEmbeds.close()
fileOutLabels.close()
