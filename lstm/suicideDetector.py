import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import sys
import torch.utils.data as data
from sklearn.metrics import classification_report

#featureGenerationMethod = sys.argv[1]
#featureGenerationMethod = "TFUDF"
#featureGenerationMethod ="LDA"
#featureGenerationMethod = "LSI"
#featureGenerationMethod = "Doc2Vec"
featureGenerationMethod = "BERT"

target_names = ['suicide', 'not-suicide']
suicideIdeation = []
notSuicideIdeation = []
training_data = []

with open("../sentimentAnalisis/suicideIdeationSentiments.obj","rb") as sf:
    suicideIdeation = pickle.load(sf)

with open("../sentimentAnalisis/NotSuicideIdeationSentiments.obj","rb") as nsf:
    notSuicideIdeation = pickle.load(nsf)
print("Identificando notas suicidas...")
print("Notas suicidas:{}  Notas no suicidas:{}".format(len(suicideIdeation), len(notSuicideIdeation)))

training_data = suicideIdeation + notSuicideIdeation
#print(training_data[0])
#for item in training
training_labels = [1] * len(suicideIdeation) + [0] * len(notSuicideIdeation)
flabels = []
for label in training_labels:
    if label == 0:
        flabels.append([1,0]) #Si la clase es 0 (suicida) entonces coloca el 1 en la posición 0, así argmax de [1, 0] es 0
    else:
        flabels.append([0,1]) #Si la clase es 1 (no-suicida) entonces coloca el 1 en la posición 1, así argmax de [0, 1] es 1
training_labels = flabels
#training_labels = torch.Tensor(training_labels)
training_labels2 = np.array(training_labels)
train_dataset = [] 
for vector,label in zip(training_data,training_labels):
    train_dataset.append([vector,label])



EMBEDDING_DIM = len(training_data[0][0])
HIDDEN_DIM = 10
CLASSES = 2


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, classes):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.classes = classes
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        #self.lstm.to(device)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, classes)
        #self.hidden2tag.to(device)
    def forward(self, embeds):
        self.embeds = embeds

        lstm_out, _ = self.lstm(embeds.view(len(embeds), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(embeds), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores[len(tag_scores)-1]



loss_function = nn.CrossEntropyLoss()
# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    #model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        for idx in range(len(data)):
            dataIn, targetIn = data[idx].to(device), target[idx].to(device)
            optimizer.zero_grad()
            output = model(dataIn)
            loss = loss_function(output, targetIn)
            loss.backward()
            optimizer.step()


# Define the number of folds and batch size
k_folds = 5
batch_size = 64

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Initialize the k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True)

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.Tensor(target)
    return [data, target]
classificationReportFile = open("../classificationReport/"+featureGenerationMethod+"-classificationReport.txt","w")
classificationReportFile.write(featureGenerationMethod+"\n")
# Loop through each fold
for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")
    print("-------")

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=my_collate,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=my_collate,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    # Initialize the model and optimizer
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # Train the model on the current fold
    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    correct = 0
    dataCounter = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            dataCounter = dataCounter + len(data)
            for idx in range(len(data)):
                dataIn, targetIn = data[idx].to(device), target[idx].to(device)
                output = model(dataIn)
                test_loss += loss_function(output, targetIn).item()
                pred = torch.argmax(output)
                preds.append(pred.item())
                targets.append(torch.argmax(targetIn).item())
                correct += pred.eq(torch.argmax(target[idx])).sum().item()
    fold_report = classification_report(targets, preds, target_names=target_names,digits=5)
    classificationReportFile.write(fold_report+"\n")
    print(fold_report)

    test_loss /= len(test_loader.dataset)
classificationReportFile.close()
#    accuracy = 100.0 * correct / len(test_loader.dataset)
#    accuracy2 = 100.0 * correct / dataCounter

    # Print the results for the current fold
#    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
#    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{dataCounter} ({accuracy2:.2f}%)\n")
