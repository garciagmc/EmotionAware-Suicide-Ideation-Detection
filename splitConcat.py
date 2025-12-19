import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

#SEED = 42
SEED = 45
K_FOLDS = 5


print("Creating splits from BERT + Emotions")

with open("adaptData/concat/Suicide_BERT_Emotions.obj", "rb") as sf:
    suicideIdeation = pickle.load(sf)

with open("adaptData/concat/NotSuicide_BERT_Emotions.obj", "rb") as nsf:
    notSuicideIdeation = pickle.load(nsf)

print(f"suicide: {len(suicideIdeation)}")
print(f"not suicide: {len(notSuicideIdeation)}")

training_data = suicideIdeation + notSuicideIdeation

training_labels = []
for _ in suicideIdeation:
    training_labels.append("suicide")
for _ in notSuicideIdeation:
    training_labels.append("not-suicide")

#print(f"\n2. Total samples: {len(training_data)}")
#print(f"   Feature dimension: {training_data[0].shape}")
#print(f"   Unique classes: {set(training_labels)}")


data = {
    "features": training_data,
    "labels": training_labels,
    "suicide_count": len(suicideIdeation),
    "not_suicide_count": len(notSuicideIdeation)
}

# Crear K-Fold splits
print(f"\n Creating {K_FOLDS} folds con K-Fold Cross Validation...")
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
splits = {}

for fold, (train_idx, test_idx) in enumerate(kf.split(training_data), 1):
    splits[f"fold_{fold}"] = {
        "train": train_idx.tolist(),
        "test": test_idx.tolist()
    }
    
    # Check distribution in train
    train_labels = [training_labels[i] for i in train_idx]
    train_suicide = train_labels.count("suicide")
    train_not_suicide = train_labels.count("not-suicide")
    
    # Check distribution in test
    test_labels = [training_labels[i] for i in test_idx]
    test_suicide = test_labels.count("suicide")
    test_not_suicide = test_labels.count("not-suicide")
    
    print(f"   Fold {fold}:")
    print(f"      Train: {len(train_idx)} samples (suicide={train_suicide}, not-suicide={train_not_suicide})")
    print(f"      Test:  {len(test_idx)} samples (suicide={test_suicide}, not-suicide={test_not_suicide})")


#print("Saving data and splits")
os.makedirs("reproducibilidad", exist_ok=True)

with open("reproducibilidad/features_data.pkl", "wb") as f:
    pickle.dump(data, f)
#print(" Features saved at: reproducibility/features_data.pkl")

with open("reproducibilidad/kfold_splits.pkl", "wb") as f:
    pickle.dump(splits, f)
print("Splits saved at: reproducibility/kfold_splits.pkl")


metadata = {
    "seed": SEED,
    "k_folds": K_FOLDS,
    "total_samples": len(training_data),
    "suicide_samples": len(suicideIdeation),
    "not_suicide_samples": len(notSuicideIdeation),
    "feature_dim": training_data[0].shape[1] if hasattr(training_data[0], 'shape') else len(training_data[0]),
    "data_source": "BERT+Emotions concatenated features",
    "source_files": [
        "../adaptData/concat/Suicide_BERT_Emotions.obj",
        "../adaptData/concat/NotSuicide_BERT_Emotions.obj"
    ]
}

with open("reproducibilidad/splits_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Seed used: {SEED}")
print(f"Number of folds: {K_FOLDS}")
print(f"The splits correspond to BERT+Emotions features")