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


#dataset=pd.read_csv("../adaptData/emotionsCEASE/D2Vectors.csv", sep=';')


embedsFile = open("../adaptData/emotionsCEASE/embeds.obj","rb")
labelsFile = open("../adaptData/emotionsCEASE/labels.obj","rb")

#suicideIdeationEmbedsFile = open("../adaptData/suicide/embeds.obj","rb")

embeds = np.array(pickle.load(embedsFile))
labels = pickle.load(labelsFile)
print(len(embeds))
print(f"Tamaño del vector {len(embeds[0])}")
print(len(np.unique(labels)))

 #Comprobaciones iniciales
print(f"Cantidad de embeddings: {len(embeds)}")
print(f"Tamaño de cada embedding: {len(embeds[0])}")
print(f"Etiquetas únicas: {np.unique(labels)}")

#for item in embeds:
#    print(item)
#X, y = sklearn.datasets.make_circles(n_samples=1001, noise=0.2, factor=0.25)
inputDim = len(embeds[0])
hiddenDim = 50
outputDim =  15

newLabels = []


for label in labels:
    newLabel = [0] * outputDim
    newLabel[label] = 1
    newLabels.append(newLabel)
    #print(newLabel)
labels = np.array(newLabels)

#labels = np.array(labels) 
#print(labels[800])



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

model = MultiLayerPerceptron(input_dim=inputDim, output_dim=outputDim, hidden_dim=hiddenDim)


class MyDataset(data.Dataset):
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data.astype('float32'))
        self.labels = torch.from_numpy(labels.astype('float32'))
        
    def __getitem__(self, idx: int):
        return (self.data[idx], self.labels[idx])
    
    def __len__(self):
        return len(self.labels)
    
    
dataset = MyDataset(embeds, labels)
dataset_size = len(dataset)

#train_set, valid_set, test_set = data.random_split(dataset, [600, 100, 95], 
                                                   #generator=torch.Generator().manual_seed(34))


# Ajustar proporciones para dividir el dataset
train_size = int(dataset_size * 0.7)  # 70% para entrenamiento
valid_size = int(dataset_size * 0.2)  # 20% para validación
test_size = dataset_size - (train_size + valid_size)  # El resto para prueba

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_set, valid_set, test_set = data.random_split(dataset, [train_size, valid_size, test_size], 
                                                   generator=torch.Generator().manual_seed(34))

# Verificar los tamaños
print(f"Tamaño total del dataset: {dataset_size}")
print(f"Entrenamiento: {len(train_set)}, Validación: {len(valid_set)}, Prueba: {len(test_set)}")


train_loader = data.DataLoader(train_set, shuffle=True, batch_size=64)
valid_loader = data.DataLoader(valid_set, shuffle=False, batch_size=64)
test_loader = data.DataLoader(test_set, shuffle=False, batch_size=64)

def update_step(data, label):
    prediction = model(data)
    #print(prediction)
    optimizer.zero_grad()
    loss = criterion(prediction, label)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_step(data, label):
    prediction = model(data)
    loss = criterion(prediction, label)
    return loss.item()

def train_one_epoch(epoch):    
    train_loss, valid_loss = 0.0, 0.0    
    for batchx, batchy in train_loader:
        train_loss += update_step(batchx, batchy)    
    for batchx, batchy in valid_loader:
        valid_loss += evaluate_step(batchx, batchy)
        
    # Guardar modelo si es el mejor hasta ahora
    global best_valid_loss
    if epoch % 10 == 0:
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'epoca': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss}, 
                       'best_model.pt')
    
    return train_loss/len(train_loader.dataset), valid_loss/len(valid_loader.dataset)



criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

max_epochs, best_valid_loss = 500, np.inf
running_loss = np.zeros(shape=(max_epochs, 2))

for epoch in range(max_epochs):
    running_loss[epoch] = train_one_epoch(epoch)
    #print(running_loss[epoch])
    
fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
ax.plot(running_loss[:, 0], label='Entrenamiento')
ax.plot(running_loss[:, 1], label='Validación')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend();
#plt.show()


#torch.save(net.state_dict(),"emotionsNet.pt")

saved_model = torch.load('best_model.pt')
print("modelo guardado")
print(saved_model['epoca'])

#model = my_model(5)
model.load_state_dict(saved_model['model_state_dict'])


torch.save(model,"sentimentModel.pt")

ytrue, ypred = [], []
for x, y_ in test_loader:
    ypred.append(nn.Softmax(dim=1)(model(x)).detach().argmax(dim=1))
    print("tamaño {}".format(len(x[0])))
    ytrue.append(y_.argmax(dim=1))
    #print(y_.argmax(dim=1))

#trueValues = ([item for item in ytrue])
#print(trueValues)
#for item in ytrue:
    #print(item)
ytrue, ypred = np.concatenate(ytrue), np.concatenate(ypred)


print(classification_report(ytrue, ypred))

with open ("test_results.txt", "w") as results_file:
    results_file.write("True Labels vs Predicted Labels\n")
    results_file.write("-" * 40 + "\n")
    for true, pred in zip(ytrue, ypred):
        results_file.write(f"True: {true}, Predicted: {pred}\n")
    results_file.write("\nClassification Report: \n")
    results_file.write(classification_report(ytrue, ypred))

print ("Resultados guardados en 'test_results.txt'")
print ("Entrenamiento completado. Modelo Guardado ")


#embedsSuicideIdeation = np.array(pickle.load(suicideIdeationEmbedsFile))
#embedsSuicideIdeation = torch.FloatTensor(embedsSuicideIdeation)
#print(len(embedsSuicideIdeation))

#emotionsInSuicideFile = open("emotionsInSucideIdeationsEmbeds.obj","wb")

#emotionsInSuicidePredictions = torch.tensor(())

#for vector in embedsSuicideIdeation:
#    prediction = model(vector).view(1,13)
    #print(prediction.size())
    #print(prediction)
#    emotionsInSuicidePredictions = torch.cat((emotionsInSuicidePredictions,prediction),dim=0)

#print(emotionsInSuicidePredictions)
#print(len(emotionsInSuicidePredictions))
#pickle.dump(embeds,fileOutEmbeds)

#emotionsInSuicideFile.close()
#suicideIdeationEmbedsFile.close()
