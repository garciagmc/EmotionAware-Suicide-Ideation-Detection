#Basic setup

import logging
import multiprocessing
from pprint import pprint

import smart_open
from gensim.corpora.wikicorpus import WikiCorpus, tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#from gensim.models import LdaModel

from gensim.test.utils import get_tmpfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Preparing corpus

wiki = WikiCorpus(
    "../Corpus/enwiki-latest-pages-articles.xml.bz2",  # path to the file you downloaded above
    tokenizer_func=tokenize,  # simple regexp; plug in your own tokenizer here
    metadata=True,  # also return the article titles and ids when parsing
    dictionary={},  # don't start processing the data yet
)

with smart_open.open("wiki/wiki.txt.gz", "w", encoding='utf8') as fout:
    for article_no, (content, (page_id, title)) in enumerate(wiki.get_texts()):
        title = ' '.join(title.split())
        if article_no % 500000 == 0:
            logging.info("processing article #%i: %r (%i tokens)", article_no, title, len(content))
        fout.write(f"{title}\t{' '.join(content)}\n")  # title_of_article [TAB] words of the article

class TaggedWikiCorpus:
    def __init__(self, wiki_text_path):
        self.wiki_text_path = wiki_text_path
        
    def __iter__(self):
        for line in smart_open.open(self.wiki_text_path, encoding='utf8'):
            title, words = line.split('\t')
            yield TaggedDocument(words=words.split(), tags=[title])

documents = TaggedWikiCorpus('wiki/wiki.txt.gz')  # A streamed iterable; nothing in RAM yet.

#Traning
workers = multiprocessing.cpu_count() - 1  # leave one core for the OS & other stuff

# PV-DBOW: paragraph vector in distributed bag of words mode
model_dbow = Doc2Vec(
    dm=0, dbow_words=1,  # dbow_words=1 to train word vectors at the same time too, not only DBOW
    vector_size=300, window=8, epochs=10, workers=workers, max_final_vocab=1000000,
)

# PV-DM: paragraph vector in distributed memory mode
model_dm = Doc2Vec(
    dm=1, dm_mean=1,  # use average of context word vectors to train DM
    vector_size=300, window=8, epochs=10, workers=workers, max_final_vocab=1000000,
)


model_dbow.build_vocab(documents, progress_per=500000)
#print(model_dbow)

# Save some time by copying the vocabulary structures from the DBOW model to the DM model.
# Both models are built on top of exactly the same data, so there's no need to repeat the vocab-building step.
model_dm.reset_from(model_dbow)
#print(model_dm)

#lda = LdaModel(documents, num_topics=100)
# Train DBOW doc2vec incl. word vectors.
# Report progress every ½ hour.
model_dbow.train(documents, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs, report_delay=30*60)

# Train DM doc2vec.
model_dm.train(documents, total_examples=model_dm.corpus_count, epochs=model_dm.epochs, report_delay=30*60)

#Persist models to disk

#dbowModelFile = get_tmpfile("dbow_doc2vec_model")
#dmModelFile = get_tmpfile("dm_doc2vec_model")

model_dbow.save("model/doc2vec.dbowModel")
model_dm.save("model/doc2vec.dmModel")


#for model in [model_dbow, model_dm]:
#    print(model)
#    pprint(model.dv.most_similar(positive=["hello world"], topn=20))