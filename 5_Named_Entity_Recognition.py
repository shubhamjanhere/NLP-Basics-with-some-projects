"""
Please install the following packages by typing the following commands -
import nltk
nltk.download("maxent_ne_chunker")
nltk.download("words")
"""
"""
ne_chunk needs part-of-speech annotations to add NE labels to the sentence. The output of the ne_chunk is a nltk.Tree object.
The ne_chunk function acts as a chunker, meaning it produces 2-level trees:
            Nodes on Level-1: Outside any chunk
            Nodes on Level-2: Inside a chunk – The label of the chunk is denoted by the label of the subtree
"""
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import treebank


sentence = "Mark and John are working at Google."
print(ne_chunk(pos_tag(word_tokenize(sentence))))
#In this example, Mark/NNP is a level-2 leaf, part of a PERSON chunk. and/CC is a level-1 leaf, meaning it’s
# not part of any chunk.
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
iob_tagged = tree2conlltags(ne_tree)
print(iob_tagged)
"""
The IOB Tagging system contains tags of the form:
        B-{CHUNK_TYPE} – for the word in the Beginning chunk
        I-{CHUNK_TYPE} – for words Inside the chunk
        O – Outside any chunk
A sometimes used variation of IOB tagging is to simply merge the B and I tags:
        {CHUNK_TYPE} – for words inside the chunk
        O – Outside any chunk
"""
ne_tree = conlltags2tree(iob_tagged)        #Same as ->ne_chunk(pos_tag(word_tokenize(sentence)))
print(ne_tree)
"""
You can also perform chunking through the above module. Chunking (aka. Shallow parsing) is to analyzing a sentence to 
identify the constituents (noun groups, verbs, verb groups, etc.). However, it does not specify their internal structure, 
nor their role in the main sentence.
"""
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
print("entities = %s"%(entities))
"""
In case you want to draw parse trees - 
    nltk.download('maxent_treebank_pos_tagger')
    nltk.download('treebank')
"""
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()



"""
Please visit - http://nlpforhackers.io/named-entity-extraction/ for knowing the complete implementation of NER in projects.
"""

