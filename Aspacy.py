###  Assignment No 1 ###
#Name : Sachin Bhaskar Sadgir
#Batch : B3
#Roll No : 50
#Assignment Title : Text pre-processing using NLP operation : perform Tokenization
#Stop word removal, Punctuation removal,using Spacy or NLTK Library 

#import library
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a test sentence.")


# ######Sentence Detection#######
about_text = (
    " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
about_doc = nlp(about_text)
sentences = list(about_doc.sents)
len(sentences)

for sentence in sentences:
    print(f"{sentence[:5]}...")

# ########### TOKEN IN SPACY ###############

import spacy
nlp = spacy.load("en_core_web_sm")
about_text = (
    " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
about_doc = nlp(about_text)

for token in about_doc:
    print (token, token.idx)


print(
    f"{'Text with Whitespace':22}"
    f"{'Is Alphanumeric?':15}"
    f"{'Is Punctuation?':18}"
    f"{'Is Stop Word?'}"
)
for token in about_doc:
    print(
        f"{str(token.text_with_ws):22}"
        f"{str(token.is_alpha):15}"
        f"{str(token.is_punct):18}"
        f"{str(token.is_stop)}"
    )



########### STOP WORDS REMOVAL ##############
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
len(spacy_stopwords)

for stop_word in list(spacy_stopwords)[:10]:
    print(stop_word)


custom_about_text = (
      " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
nlp = spacy.load("en_core_web_sm")
about_doc = nlp(custom_about_text)
print([token for token in about_doc if not token.is_stop])



##################### Lemmatization #################


conference_help_text = (
      " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
conference_help_doc = nlp(conference_help_text)
for token in conference_help_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")

################# WORD FREQUENCY ############

from collections import Counter
nlp = spacy.load("en_core_web_sm")
complete_text = (
      " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
complete_doc = nlp(complete_text)

words = [
    token.text
    for token in complete_doc
    if not token.is_stop and not token.is_punct
]

print(Counter(words).most_common(5))

############### POS Tagging #############
import spacy
nlp = spacy.load("en_core_web_sm")
about_text = (
      " My name is sachin"
    " I am a undergraude student"
    " My comapany name is SachinITTech"
    " I am the founder of the company."
)
about_doc = nlp(about_text)
for token in about_doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )

"""
# ######Sentence Detection#######
My name is sachin I...

#############       Tokenization         ###############

  0
My 1
name 4
is 9
sachin 12
I 19
am 21
a 24
undergraude 26
student 38
My 46
comapany 49
name 58
is 63
SachinITTech 66
I 79
am 81
the 84
founder 88
of 96
the 99
company 103
. 110


### TOKENIZATION WITH ATTRIBUTE #############
Text with Whitespace  Is Alphanumeric?Is Punctuation?   Is Stop Word?
                      False          False             False 
My                    True           False             True  
name                  True           False             True  
is                    True           False             True  
sachin                True           False             False 
I                     True           False             True  
am                    True           False             True  
a                     True           False             True  
undergraude           True           False             False 
student               True           False             False 
My                    True           False             True  
comapany              True           False             False 
name                  True           False             True  
is                    True           False             True  
SachinITTech          True           False             False 
I                     True           False             True  
am                    True           False             True  
the                   True           False             True  
founder               True           False             False 
of                    True           False             True  
the                   True           False             True  
company               True           False             False 
.                     False          True              False 

################ STOP WORD REMOVAL ################
its
such
same
again
neither
quite
everywhere
ours
before
moreover


[ , sachin, undergraude, student, comapany, SachinITTech, founder, company, .]

######################Lemmatization####################
                 My : my
                  is : be
                  am : be
                  My : my
                  is : be
                  am : be

###################  WORD FREQUENCY #################
[(' ', 1), ('sachin', 1), ('undergraude', 1), ('student', 1), ('comapany', 1)]


################## Part-of-Speech Tagging ##############
TOKEN:
=====
TAG: _SP        POS: SPACE
EXPLANATION: whitespace

TOKEN: My
=====
TAG: PRP$       POS: PRON
EXPLANATION: pronoun, possessive

TOKEN: name
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: is
=====
TAG: VBZ        POS: AUX
EXPLANATION: verb, 3rd person singular present

TOKEN: sachin
=====
TAG: JJ         POS: ADJ
EXPLANATION: adjective (English), other noun-modifier (Chinese)

TOKEN: I
=====
TAG: PRP        POS: PRON
EXPLANATION: pronoun, personal

TOKEN: am
=====
TAG: VBP        POS: AUX
EXPLANATION: verb, non-3rd person singular present

TOKEN: a
=====
TAG: DT         POS: DET
EXPLANATION: determiner

TOKEN: undergraude
=====
TAG: JJ         POS: ADJ
EXPLANATION: adjective (English), other noun-modifier (Chinese)

TOKEN: student
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: My
=====
TAG: PRP$       POS: PRON
EXPLANATION: pronoun, possessive

TOKEN: comapany
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: name
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: is
=====
TAG: VBZ        POS: AUX
EXPLANATION: verb, 3rd person singular present

TOKEN: SachinITTech
=====
TAG: NNP        POS: PROPN
EXPLANATION: noun, proper singular

TOKEN: I
=====
TAG: PRP        POS: PRON
EXPLANATION: pronoun, personal

TOKEN: am
=====
TAG: VBP        POS: AUX
EXPLANATION: verb, non-3rd person singular present

TOKEN: the
=====
TAG: DT         POS: DET
EXPLANATION: determiner

TOKEN: founder
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: of
=====
TAG: IN         POS: ADP
EXPLANATION: conjunction, subordinating or preposition       

TOKEN: the
=====
TAG: DT         POS: DET
EXPLANATION: determiner

TOKEN: company
=====
TAG: NN         POS: NOUN
EXPLANATION: noun, singular or mass

TOKEN: .
=====
TAG: .          POS: PUNCT
EXPLANATION: punctuation mark, sentence closer
"""