import sys
import torch
import torch.nn as nn
import pandas as pd
import pickle
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import sklearn.datasets
from sklearn.metrics import classification_report


outputFileName = sys.argv[1]# Recibe el nombre del archivo de salida
embedsFilePath = sys.argv[2]# Recibe el nombre del archivo con los vectores de emociones que representan al texto suicida/no suicida

print(f"Generando distribuciones de emociones de las notas suicidas/no-suicidas...{outputFileName}")

suicideIdeationEmbedsFile = open(embedsFilePath,"rb")


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(type(self), self).__init__()  
        self.hidden = nn.Linear(input_dim, hidden_dim)
        #self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(hidden_dim, output_dim)        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        #x = self.hidden2(x)
        x = self.activation(x)
       
        return self.output(x)

model = torch.load("sentimentModel.pt")
model.eval()

embedsSuicideIdeation = pickle.load(suicideIdeationEmbedsFile)

emotionsInSuicideFile = open(outputFileName+".obj","wb")

emotionsInSuicidePredictions = torch.tensor(())
resultTensors = []
for vectorSet in embedsSuicideIdeation:
    tvectors = torch.FloatTensor(vectorSet)
    actualSetPredictions = torch.tensor(())
    for vector in tvectors:
        prediction = nn.Softmax(dim=0)(model(vector)).view(1,15).detach()
        actualSetPredictions = torch.cat((actualSetPredictions,prediction),dim=0)
    resultTensors.append(actualSetPredictions)

pickle.dump(resultTensors,emotionsInSuicideFile)

emotionsInSuicideFile.close()
suicideIdeationEmbedsFile.close()
