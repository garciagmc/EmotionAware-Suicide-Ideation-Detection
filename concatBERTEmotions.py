import pickle
import numpy as np
import torch

def concat_class(bert_embeds_path, emo_embeds_path, out_path):
    bertFile = open(bert_embeds_path, "rb")
    emoFile  = open(emo_embeds_path, "rb")

    bert_docs = pickle.load(bertFile)
    emo_docs  = pickle.load(emoFile)

    bertFile.close()
    emoFile.close()

    assert len(bert_docs) == len(emo_docs), "Number of documents does not match"

    combined_docs = []

    for bdoc, edoc in zip(bert_docs, emo_docs):

        # Emotions come as a tensor.
        if isinstance(edoc, torch.Tensor):
            edoc = edoc.numpy()

        assert len(bdoc) == len(edoc), "Number of sentences does not match"

        combined_doc = []
        for b, e in zip(bdoc, edoc):
            combined_doc.append(np.concatenate((b, e), axis=0))

        combined_docs.append(np.array(combined_doc))

    outFile = open(out_path, "wb")
    pickle.dump(combined_docs, outFile)
    outFile.close()

    print(f"File generated: {out_path}")

   # print(f"{out_path} → documents:", len(combined_docs))

concat_class(
    bert_embeds_path="adaptData/suicide/SuicideembedsSent.obj",
    emo_embeds_path ="sentimentAnalisis/suicideIdeationSentiments.obj",
    out_path        ="adaptData/concat/Suicide_BERT_Emotions.obj"
)

concat_class(
    bert_embeds_path="adaptData/suicide/NotSuicideembedsSent.obj",
    emo_embeds_path ="sentimentAnalisis/NotSuicideIdeationSentiments.obj",
    out_path        ="adaptData/concat/NotSuicide_BERT_Emotions.obj"
)
