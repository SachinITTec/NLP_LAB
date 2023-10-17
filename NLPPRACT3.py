# sachin sadgir 
# Batch -B3
# Subject :-NLP


import spacy
raw_text="""Certainly, here's a paragraph with various named entities (NEs) highlighted:

"In recent years, **Apple Inc.** has revolutionized the technology industry with groundbreaking products like the **iPhone**, **MacBook Pro**, and the **Apple Watch**. As a **final-year student**, **Sachin** is determined to secure a bright future in the world of **engineering** and **B.Tech**. He aspires to work for renowned companies such as **Tesla**, **Microsoft**, or **Amazon**. Sachin's academic prowess and leadership skills have earned him recognition at his university, **Stanford**, and he is a proud member of the **IEEE** student chapter. In addition to his academic commitments, Sachin enjoys unwinding with friends at his favorite local hangout, **Starbucks**, and he's an avid fan of the **New York Yankees** baseball team."

This paragraph contains a variety of named entities, including organizations (Apple Inc., Tesla, Microsoft, Amazon, IEEE, Stanford), products (iPhone, MacBook Pro, Apple Watch), a person's name (Sachin), educational institutions (Stanford), a professional qualification (B.Tech), a location (Starbucks), and a sports team (New York Yankees). """
NER = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
text= NER(raw_text)
for w in text.ents:
    print(w.text,w.label_)
spacy.displacy.render(text, style="ent",jupyter=True)


# recent years DATE
# Apple Inc. ORG
# Sachin PERSON
# Sachin PERSON
# New York Yankees GPE
# Apple Inc. ORG
# Tesla ORG
# Microsoft ORG
# Amazon ORG
# IEEE ORG
# Stanford ORG
# MacBook Pro PERSON
# Apple Watch ORG
# Stanford ORG
# B.Tech ORG
# Starbucks ORG
# New York Yankees ORG
# <IPython.core.display.HTML object>'''
