import pandas as pd

dataset=pd.read_csv("../Corpora/emotions/CEASEBalanced.csv", sep=';')

sentences = dataset.iloc[:,0]
labels = dataset.iloc[:,1]

sentencesFile = open("emotionsCEASE/sentencesCEASE.txt","w")
labelsFile = open("emotionsCEASE/labelsCEASE.txt","w")

for sent in sentences:
	sentencesFile.write(sent+"\n")

for label in labels:
	labelsFile.write(label+"\n")

sentencesFile.close()
labelsFile.close()