import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

featureGenerationMethod = "BERT"
target_names = ['suicide', 'not-suicide']

print("LSTM with reproducibiity")

#print("Loading BERT+Emotions features")
with open("../adaptData/concat/Suicide_BERT_Emotions.obj","rb") as sf:
    suicideIdeation = pickle.load(sf)

with open("../adaptData/concat/NotSuicide_BERT_Emotions.obj","rb") as nsf:
    notSuicideIdeation = pickle.load(nsf)

training_data = suicideIdeation + notSuicideIdeation

#Labels
training_labels = []
for _ in suicideIdeation:
    training_labels.append([0,1])  
for _ in notSuicideIdeation:
    training_labels.append([1,0])  

train_dataset = []
for vector, label in zip(training_data, training_labels):
    train_dataset.append([vector, label])

EMBEDDING_DIM = training_data[0].shape[1]
HIDDEN_DIM = 32
CLASSES = 2

print("Loading pre-saved splits")
with open("../reproducibilidad/kfold_splits.pkl", "rb") as f:
    pre_saved_splits = pickle.load(f)

with open("../reproducibilidad/splits_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f" Splits loaded")
print(f" Number of folds: {len(pre_saved_splits)}")
print(f" Seed used: {metadata['seed']}")
print(f" Total samples: {metadata['total_samples']}")
print(f" Feature dimension: {metadata['feature_dim']}")


# Verify that the data matches metadata
assert len(train_dataset) == metadata['total_samples'], \
    f"ERROR: Sample size mismatch. Dataset={len(train_dataset)}, Metadata={metadata['total_samples']}"

assert EMBEDDING_DIM == metadata['feature_dim'], \
    f"ERROR: Feature dimension mismatch. Dataset={EMBEDDING_DIM}, Metadata={metadata['feature_dim']}"

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        b, s, e = x.size()
        x = x.view(b, s, self.num_heads, self.head_dim)
        return x.permute(0,2,1,3)

    def combine_heads(self, x):
        b, h, s, d = x.size()
        x = x.permute(0,2,1,3).contiguous()
        return x.view(b, s, h*d)

    def forward(self, x):
        Q = self.split_heads(self.q(x))
        K = self.split_heads(self.k(x))
        V = self.split_heads(self.v(x))

        scores = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        out = self.combine_heads(out)
        return self.out(out), weights

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attn(x)

class LSTMTaggerWithAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, classes, num_heads):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.attention = AttentionLayer(hidden_dim, num_heads)
        self.hidden2tag = nn.Linear(hidden_dim, classes)

    def forward(self, embeds):
        lstm_out, _ = self.lstm(embeds.view(len(embeds), 1, -1))
        attn_out, _ = self.attention(lstm_out)
        tag_space = self.hidden2tag(attn_out[:, -1, :])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores[-1]

loss_function = nn.NLLLoss()

def my_collate(batch):
    data = [torch.FloatTensor(item[0]) for item in batch]
    target = torch.LongTensor([np.argmax(item[1]) for item in batch])
    return data, target

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        for doc, label in zip(data, target):
            doc = doc.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(doc)
            loss = loss_function(output.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()


batch_size = 64
device = torch.device("cpu")
num_epochs = 100

os.makedirs("../classificationReport/TrainReports", exist_ok=True)
os.makedirs("../checkpoints", exist_ok=True)

#Report
classificationReportFile = open(
    "../classificationReport/"+featureGenerationMethod+"-classificationReport.txt","w"
)
classificationReportFile.write(featureGenerationMethod+"\n")

classificationReportFileTrain = open(
    "../classificationReport/TrainReports/"+featureGenerationMethod+"-classificationReportTrain.txt","w"
)
classificationReportFileTrain.write(featureGenerationMethod+"\n")

#Cross-validation with pre-saved splits
for fold_name in sorted(pre_saved_splits.keys()):
    fold_num = int(fold_name.split('_')[1])
    train_idx = pre_saved_splits[fold_name]["train"]
    test_idx = pre_saved_splits[fold_name]["test"]
    
    print(f"FOLD {fold_num}/{len(pre_saved_splits)}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    torch.manual_seed(SEED + fold_num)
    np.random.seed(SEED + fold_num)
    random.seed(SEED + fold_num)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        collate_fn=my_collate
    )

    test_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        collate_fn=my_collate
    )

    model = LSTMTaggerWithAttention(
        EMBEDDING_DIM, HIDDEN_DIM, CLASSES, num_heads=8
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    #print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        if epoch % 20 == 0:
           # print(f"  Epoch {epoch}/{num_epochs} completed")

    # Save checkpoint
    checkpoint_path = f"../checkpoints/{featureGenerationMethod}_seed{SEED}_fold{fold_num}.pt"
    torch.save(
       {
           "seed": SEED,
           "fold": fold_num,
           "train_indices": train_idx,
           "test_indices": test_idx,
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
           "using_presaved_splits": True,
           "embedding_dim": EMBEDDING_DIM,
           "hidden_dim": HIDDEN_DIM,
           "num_heads": 8,
           "epochs_trained": num_epochs,
           "metadata": metadata
       },
       checkpoint_path
    )

    #Evaluating on training set
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            for doc, label in zip(data, target):
                output = model(doc.to(device))
                preds.append(torch.argmax(output).item())
                targets.append(label.item())

    report_train = classification_report(targets, preds, target_names=target_names, digits=5)
    #ga_train = core.rga(targets, preds)
    auc_train = roc_auc_score(targets, preds)
    
    classificationReportFileTrain.write(report_train)
    #classificationReportFileTrain.write("RGA-AUC"+"\t"+str(rga_train)+"\t"+str(auc_train)+"\n\n")

    print("\nTRAIN RESULTS:")
    print(report_train)
    #print(f"RGA: {rga_train}")
    print(f"AUC: {auc_train:.5f}")

    print("\nEvaluando en conjunto de prueba...")
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            for doc, label in zip(data, target):
                output = model(doc.to(device))
                preds.append(torch.argmax(output).item())
                targets.append(label.item())

    report_test = classification_report(targets, preds, target_names=target_names, digits=5)
    #rga_test = core.rga(targets, preds)
    auc_test = roc_auc_score(targets, preds)
    
    classificationReportFile.write(report_test)
    #classificationReportFile.write("RGA-AUC"+"\t"+str(rga_test)+"\t"+str(auc_test)+"\n\n")

    print("\nTEST RESULTS:")
    print(report_test)
    #print(f"RGA: {rga_test}")
    print(f"AUC: {auc_test:.5f}")

classificationReportFile.close()
classificationReportFileTrain.close()
