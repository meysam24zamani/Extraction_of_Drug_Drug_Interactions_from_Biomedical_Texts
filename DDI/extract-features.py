#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
#stem
from nltk.stem import PorterStemmer
ps = PorterStemmer()
#lemmatize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#stopwords
from nltk.corpus import stopwords
#bigrams
from nltk.util import bigrams
#POS
from nltk import pos_tag

######## Download resources - only once

from nltk import download
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')

#########################################

#count values in dict
from collections import Counter
#from sklearn.feature_extraction.text import CountVectorizer

## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
    return tks

def reduce_tokens(txt):
    token_words = [tup[0] for tup in tokens] #only tokens
    token_words = [tok.lower() for tok in token_words] #lowercase
    token_words = [tok for tok in token_words if tok.isalpha() and tok not in stopwords.words('english')] #drop: non-words, stopwords
    return token_words

def reduce_words(strings):
    #first lemmatizing
    lemmas = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in strings]
    #then stemming
    stems = [ps.stem(word) for word in lemmas]
    return stems

def pos(inputString):
    tag = pos_tag(word_tokenize(inputString))[0][1]

    if tag.startswith('J'):
        return 'ADJ'
    elif tag.startswith('N'):
        return 'N'
    elif tag.startswith('R'):
        return 'ADV'
    elif tag.startswith('V'):
        return 'V'
    else: return tag

## ----------Get interaction------------
## -- Get pair interaction and its type

def get_interaction(p) :
   ## Complete this function to return a pair (boolean, ddi_type)
   ## depending on whether there is an interaction between e1 and e2
   ddi = p.attributes["ddi"].value
   if ddi == "true" and len(p.attributes) == 5:  ## to assert that there'a always a "type" attribute, and avoid Key Error
      ddi_type = p.attributes["type"].value
      return True,ddi_type
   return False,"null"

## --------- Sentence feature extractor -----------
## -- Extract features for each sentence

def extract_features_from_sentence(s,tokens,token_words_reduced,bigrams,entities) :
   # for each sentence, generate list of features and add it to the result
   sentenceFeatures = []

   sentenceFeatures.append("n_words="+str(len(tokens))) # number of words
   sentenceFeatures.append("n_drugs="+str(len(entities))) # number of drugs

   mto_list = []
   for t in token_words_reduced:
       if t in words_mto and t not in sentenceFeatures:
           mto_list.append(t)          # words appearing >1
#   mto_vectorizer = CountVectorizer()
#   mto_vec = mto_vectorizer.fit_transform(mto_list)
   sentenceFeatures.append("words_mto="+str(len(mto_list)))

#   bigram_list = []
#   for b in bigrams:
#       bx = '-'.join(b) #change formatting
#       if b in words_bigs_mto and bx not in sentenceFeatures:
#           bigram_list.append(bx)          # bigrams appearing >1
#   bigram_vectorizer = CountVectorizer()
#   bigs_vec = bigram_vectorizer.fit_transform(bigram_list)
#   sentenceFeatures.append("bigrams="+str(len(bigram_list)))
    ### these two variables could be collinear!!!
#	if bigram in words_bigs_mto then also word in words_mto - choose one!

   return sentenceFeatures

## --------- Pair feature extractor -----------
## -- Extract features for each pair


def extract_features_from_pair(tokens,entities,id_e1,id_e2) :

   # for each pair, generate list of features and add it to the result
   pairFeatures = [];

   e1_end = int(entities[id_e1][1]) if entities[id_e1][1].isnumeric() else 0
   e2_start = int(entities[id_e2][0]) if entities[id_e2][0].isnumeric() else 0
   words_before_first_drug = [t for t in tokens if t[1] < e1_end and t[0].isalpha()]
   words_between_drugs = [t for t in tokens if t[1] > e1_end and t[2] < e2_start and t[0].isalpha()]
   words_after_second_drug = [t for t in tokens if t[2] > e2_start and t[0].isalpha()]
   
   pairFeatures.append("n_words_before_first_drug="+str(len(words_before_first_drug)))
   pairFeatures.append("n_words_between_main_drugs="+str(len(words_between_drugs)))
   pairFeatures.append("n_words_after_second_drug="+str(len(words_after_second_drug)))
   
   
   drugs_between_drugs = 0
   for e, offs in entities.items():
       if offs[0].isnumeric() and offs[1].isnumeric(): # avoid errors in offset extraction - there are mistakes in files :(
           if int(offs[0]) > e1_end and int(offs[1]) < e2_start:
               drugs_between_drugs += 1
   pairFeatures.append("n_drugs_between_main_drugs="+str(drugs_between_drugs))
   
   if len(words_before_first_drug) > 0:
       pos_tags = [pos(w[0]) for w in words_before_first_drug]
       pairFeatures.append("n_verbs_before_first_drug="+str(pos_tags.count('V')))
       pairFeatures.append("n_adjectives_before_first_drug="+str(pos_tags.count('ADJ')))
       pairFeatures.append("n_nouns_before_first_drug="+str(pos_tags.count('N')))
       pairFeatures.append("n_adverbs_between_main_drugs="+str(pos_tags.count('ADV')))       
   else:
       pairFeatures.append("n_verbs_before_first_drug=0")
       pairFeatures.append("n_adjectives_before_first_drug=0")
       pairFeatures.append("n_nouns_before_first_drug=0")
       pairFeatures.append("n_adverbs_before_first_drug=0")  
	   
   if len(words_between_drugs) > 0:
       pos_tags = [pos(w[0]) for w in words_between_drugs]
       pairFeatures.append("n_verbs_between_main_drugs="+str(pos_tags.count('V')))
       pairFeatures.append("n_adjectives_between_main_drugs="+str(pos_tags.count('ADJ')))
       pairFeatures.append("n_nouns_between_main_drugs="+str(pos_tags.count('N')))
       pairFeatures.append("n_adverbs_between_main_drugs="+str(pos_tags.count('ADV')))       
   else:
       pairFeatures.append("n_verbs_between_main_drugs=0")
       pairFeatures.append("n_adjectives_between_main_drugs=0")
       pairFeatures.append("n_nouns_between_main_drugs=0")
       pairFeatures.append("n_adverbs_between_main_drugs=0")       

   if len(words_after_second_drug) > 0:
       pos_tags = [pos(w[0]) for w in words_after_second_drug]
       pairFeatures.append("n_verbs_after_second_drug="+str(pos_tags.count('V')))
       pairFeatures.append("n_adjectives_after_second_drug="+str(pos_tags.count('ADJ')))
       pairFeatures.append("n_nouns_after_second_drug="+str(pos_tags.count('N')))
       pairFeatures.append("n_adverbs_after_second_drug="+str(pos_tags.count('ADV')))       
   else:
       pairFeatures.append("n_verbs_after_second_drug=0")
       pairFeatures.append("n_adjectives_after_second_drug=0")
       pairFeatures.append("n_nouns_after_second_drug=0")
       pairFeatures.append("n_adverbs_after_second_drug=0")       

   return pairFeatures



## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

# directory with files to process
datadir = sys.argv[1]
#datadir = 'D:/Justyna/all/AGH/FIB/AHLT/lab/data/Test-NER/DrugBank'


##############################################################################
#first loop - collect and count all words and bigrams in database
words = []
words_bigrams = []
for f in listdir(datadir) :
    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        stext = s.attributes["text"].value   # get sentence text
        tokens = tokenize(stext)
        token_words = reduce_tokens(tokens)
        tkw_reduced = reduce_words(token_words)
        #bigrams
        bigs = list(bigrams(tkw_reduced))
        words_bigrams.append(bigs)
        #words
        words.append(tkw_reduced)
words = sum(words, [])
words_bigrams = sum(words_bigrams, [])

words_cnt = Counter(words)
words_mto = [key for key, value in words_cnt.items() if value >1] #>1 appearance

words_bigs_cnt = Counter(words_bigrams)
words_bigs_mto = [key for key, value in words_bigs_cnt.items() if value >1] #>1 appearance
##############################################################################


# process each file in directory
for f in listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text

        tokens = tokenize(stext)
        token_words = reduce_tokens(tokens)
        words_reduced = reduce_words(token_words)
        bigs = list(bigrams(words_reduced))

        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")
           entities[id] = offs

        features_s = extract_features_from_sentence(s,tokens,words_reduced,bigs,entities)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
           is_ddi, ddi_type = False, "null"
           id_e1 = p.attributes["e1"].value
           id_e2 = p.attributes["e2"].value

           features_p = extract_features_from_pair(tokens,entities,id_e1,id_e2)
           is_ddi, ddi_type = get_interaction(p)
           ddi = "1" if is_ddi else "0"

           print (sid, id_e1, id_e2, ddi, ddi_type, "|".join(str(i) for i in features_s),"|".join(str(i) for i in features_p),sep="|")