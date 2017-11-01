from stemming.porter2 import stem
import snowballstemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer #Type nltk.download('wordnet')
import nltk
from nltk.tokenize import word_tokenize
"""
from nltk.stem.api import StemmerI
from nltk.stem.regexp import RegexpStemmer
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.rslp import RSLPStemmer
"""
wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = snowballstemmer.stemmer('english')
nltk_snowball_stemmer = nltk.stem.SnowballStemmer('english')
porter_stemmer = PorterStemmer()

print(stem("fairly <-porter"))
print(porter_stemmer.stem("fairly <- from NLTK")) #Wrong Choice
print(snowball_stemmer.stemWord("fairly"))
print(nltk_snowball_stemmer.stem("fairly"))
print(lancaster_stemmer.stem("fairly"))
print(snowball_stemmer.stemWords(["fairly"]))
print(wordnet_lemmatizer.lemmatize('fairly')) #Takes time

s = "This is a simple sentence comming from a booklet. cats catlike catty cat"
tokens = word_tokenize(s)  # Generate list of tokens
tokens_pos = nltk.pos_tag(tokens) # nltk.download('averaged_perceptron_tagger') or nltk.download('all') for all data
print([stem(word) for word in tokens ])
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
print(tokens_pos)


def get_tokens():
    with open('stem_sample.txt') as stem:
        tokens = nltk.word_tokenize(stem.read())
    return tokens


def do_stemming(filtered):
    stemmed = []
    for f in filtered:
        stemmed.append(PorterStemmer().stem(f))
    # stemmed.append(LancasterStemmer().stem(f))
    # stemmed.append(SnowballStemmer('english').stem(f))
    return stemmed


tokens = get_tokens()

print(' '.join(tokens))
stemmed_tokens = do_stemming(tokens)
print(' '.join(stemmed_tokens))

result = dict(zip(tokens, stemmed_tokens))
print(result)

