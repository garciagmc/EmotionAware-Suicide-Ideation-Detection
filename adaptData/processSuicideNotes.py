import pandas as pd
flag = 1 #1-suicide 2-Not suicide

dataset=pd.read_csv("../Corpora/suicide/Suicidal_intention.csv", sep='#')

sentences = dataset.iloc[:,0]
labels = dataset.iloc[:,1]

if flag == 1:
	sentencesFile = open("suicide/sentencesSuicideIdeation.txt","w")
	labelsFile = open("suicide/labelsSuicideIdeation.txt","w")
else:
	sentencesFile = open("suicide/sentencesNotSuicideIdeation.txt","w")
	labelsFile = open("suicide/labelsNotSuicideIdeation.txt","w")

for sent in sentences:
	sentencesFile.write(sent+"\n")

for label in labels:
	labelsFile.write(label+"\n")

sentencesFile.close()
labelsFile.close()