

import gensim
import pprint
from gensim import corpora
from gensim.utils import simple_preprocess

doc_list = [
   "Hello, my name is sachin?", "I am a professional It engineering?", 
   " I am currently a student ? currently i am pursuing b.tech degree in scoe?"
]

doc_tokenized = [simple_preprocess(doc) for doc in doc_list]

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
print(BoW_corpus)

id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
print(id_words)