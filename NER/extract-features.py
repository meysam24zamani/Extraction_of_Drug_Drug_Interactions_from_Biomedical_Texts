#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import re

## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

## ----

def hasdigit(inputString):
    return bool(re.search(r'\d', inputString))
 
def hasdash(inputString):
    return any(char == '-' for char in inputString)

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]

      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      tokenFeatures.append("length:"+str(len(t)))
      if (t.isupper()) : tokenFeatures.append("isUpper")
      if (t.istitle()) : tokenFeatures.append("isTitle")
      if (t.isdigit()) : tokenFeatures.append("isDigit")
      if (hasdigit(t)) : tokenFeatures.append("hasDigit")
      if (hasdash(t)) : tokenFeatures.append("hasDash")
      

      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         tokenFeatures.append("lengthPrev:"+str(len(tPrev)))
         if (tPrev.isupper()) : tokenFeatures.append("isUpperPrev") ## I think should be: tPrev
         if (tPrev.istitle()) : tokenFeatures.append("isTitlePrev")
         if (tPrev.isdigit()) : tokenFeatures.append("isDigitPrev")
         if (hasdigit(tPrev)) : tokenFeatures.append("hasDigitPrev")
      else :
         tokenFeatures.append("BoS")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         tokenFeatures.append("lengthNext:"+str(len(tNext)))
         if (tNext.isupper()) : tokenFeatures.append("isUpperNext") ## I think should be: tNext
         if (tNext.istitle()) : tokenFeatures.append("isTitleNext")
         if (tNext.isdigit()) : tokenFeatures.append("isDigitNext")
         if (hasdigit(tNext)) : tokenFeatures.append("hasDigitNext")
      else:
         tokenFeatures.append("EoS")

      result.append(tokenFeatures)

   return result

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

# directory with files to process
datadir = sys.argv[1]
#datadir = "D:/Justyna/all/AGH/FIB/AHLT/lab/data/Train/DrugBank" #train dataset
#datadir = "D:/Justyna/all/AGH/FIB/AHLT/lab/data/Test-NER/DrugBank"


# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
