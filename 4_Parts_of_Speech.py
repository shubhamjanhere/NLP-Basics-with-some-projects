"""
First, type following commands on console- 
                import nltk
                nltk.download("tagsets")
                nltk.download("treebank")

SPACY IMPLEMENTATION - 
nlp=spacy.load('en')
sentence="Ashok killed the snake with a stick"
for token in nlp(sentence):
   print(token,token.pos_)
"""
#Show all the basic functionality of POS tagger

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

s = "This is a simple sentence"
tokens = word_tokenize(s)
"""
The process of classifying words into their parts of speech and labeling them accordingly is known as part-of-speech 
tagging, POS-tagging, or simply tagging. Parts of speech are also known as word classes or lexical categories. The 
collection of tags used for a particular task is known as a tagset. Our emphasis in this chapter is on exploiting tags, 
and tagging text automatically.
"""
tokens_pos = pos_tag(tokens)
print(tokens_pos)                   #[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('simple', 'JJ'), ('sentence', 'NN')]
text = word_tokenize("They refuse to permit us to obtain the refuse permit")
print(pos_tag(text))
"""
[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]

Notice that refuse and permit both appear as a present tense verb (VBP) and a noun (NN). E.g. refUSE is a verb meaning 
"deny," while REFuse is a noun meaning "trash" (i.e. they are not homophones). Thus, we need to know which word is being 
used in order to pronounce the text correctly. (For this reason, text-to-speech systems usually perform POS-tagging.)
Many words, like ski and race, can be used as nouns or verbs with no difference in pronunciation. 
"""
text = word_tokenize("And now for something completely different")
print(pos_tag(text))
"""
[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
Here we see that and is CC, a coordinating conjunction; now and completely are RB, or adverbs; for is IN, a preposition;
something is NN, a noun; and different is JJ, an adjective.
"""

text = nltk.word_tokenize("This is a very big sentence to show applications of POS. Naruto is fishing some fishes.")
print(text)                                     # ['Dive', 'into', 'NLTK', ':', 'Part-of-speech', 'tagging', 'and', 'POS', 'Tagger']
print(nltk.pos_tag(text))                       # [('Dive', 'JJ'), ('into', 'IN'), ('NLTK', 'NNP'), (':', ':'), ............]
# NLTK provides documentation for each tag, which can be queried using the tag, e.g., nltk.help.upenn_tagset('RB'), or a
# regular expression, e.g., nltk.help.upenn_brown_tagset('NN.*'):
print(nltk.help.upenn_tagset("JJ"))             #JJ: adjective or numeral, ordinal, ......................
print(nltk.help.upenn_tagset("IN"))             #IN: preposition or conjunction, subordinating, ..........
print(nltk.help.upenn_tagset("NNP"))            #NNP: noun, proper, singular, ............................
# Representing Tagged Tokens
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
sent = ''' The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN other/AP topics/NNS ,/, AMONG/IN them/PPO 
the/AT Atlanta/NP and/CC Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS said/VBD ``/`` 
ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT 
best/JJT interest/NN of/IN both/ABX governments/NNS ''/'' ./. '''
print([nltk.tag.str2tuple(t) for t in sent.split()])


"""
Train a TnT POS Tagger Model - This will take time!!! (-_-)
We use the first 3000 treebank tagged sentences as the train_data, and last 914 tagged sentences as the test_data, 
now we train TnT POS Tagger by the train_data and evaluate it by the test_data:
"""
"""
from nltk.corpus import treebank
from nltk.tag import tnt

print(len(treebank.tagged_sents()))             #3914
train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]
print(train_data[0])
print(test_data[0])

# Takes time
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)
print(tnt_pos_tagger.evaluate(test_data))
#0.8755881718109216
"""
"""
For saving the file-
import pickle
file_var = open('tnt_treebank_pos_tagger.pickle', 'w')
pickle.dump(tnt_pos_tagger, file_var)
file_var.close()
"""

exampleArray = ["The incredibly intimidating NLP scares people away who are not brave enough. Quite Flashy, ins't it??"]
def processContent():
    try:
        for item in exampleArray:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>}"""
            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))

processContent()

"""

"""