"""
NOTE -- A lot of code given will not match the online tutorial, since the older libraries have been updated. So please 
dont get confused with online tutorials, especially in case of saving and loading binary models.

For a deeper understanding, please go to - https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

There are two main training algorithms that can be used to learn the embedding from text; they are continuous bag of 
words (CBOW) and skip grams.
    Word2Vec models require a lot of text, e.g. the entire Wikipedia corpus. Nevertheless, we will demonstrate the principles 
using a small in-memory example of text.
    Specifically, each sentence must be tokenized, meaning divided into words and prepared (e.g. perhaps pre-filtered and 
perhaps converted to a preferred case).
    The sentences could be text loaded into memory, or an iterator that progressively loads text, required for very large 
text corpora.
There are many parameters on this constructor; a few noteworthy arguments you may wish to configure are:

            size:      (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to 
                       represent each token (word).
            window:    (default 5) The maximum distance between a target word and words around the target word.
            min_count: (default 5) The minimum count of words to consider when training the model; words with an 
                        occurrence less than this count will be ignored.
            workers:   (default 3) The number of threads to use while training.
            sg:        (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).

The defaults are often good enough when just getting started. If you have a lot of cores, as most modern computers do, 
I strongly encourage you to increase workers to match the number of cores (e.g. 8).

After the model is trained, it is accessible via the “wv” attribute. This is the actual word vector model in which 
queries can be made.
For getting text8, type the bellow command and then unzip-  wget http://mattmahoney.net/dc/text8.zip -P /tmp

Basically, the algorithm takes some unstructured text and learns “features” about each word. The neat thing is (apart 
from it learning the features completely automatically, without any human input/supervision!) that these features capture 
different relationships — both semantic and syntactic. This allows some (very basic) algebraic operations, like the above 
mentioned “king–man+woman=queen“. 

"""
from pathlib import Path
import os
from gensim.models import word2vec, KeyedVectors

# load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
sentences = word2vec.Text8Corpus(os.getcwd()+'/tmp/text8')
text8_bin_path = Path(os.getcwd()+"/tmp/vectors.bin")
if text8_bin_path.is_file():
    model = KeyedVectors.load_word2vec_format(os.getcwd() + '/tmp/vectors.bin', binary=True)
else:
    ## train the skip-gram model; default window=5 .This piece of code will take time (-_-)
    model = word2vec.Word2Vec(sentences, size=200, workers=4)
# Woman + Kind - Man = Queen
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
print(model.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))
print(model.doesnt_match("breakfast cereal dinner lunch".split()))
print(model.wv.similarity('woman', 'man'))
# pickle the entire model to disk, so we can load&resume training later. Store the learned weights, in a format the
# original C tool understands
model.wv.save_word2vec_format(os.getcwd()+'/tmp/vectors.bin', binary=True)
# Load the above saved model from disk
model = KeyedVectors.load_word2vec_format(os.getcwd()+'/tmp/vectors.bin', binary=True)
# If Girl = Father, Than Boy = ? =>Mother
print(model.most_similar(['girl', 'father'], ['boy'], topn=3))
more_examples = ["he his she", "big bigger bad", "going went being"]

for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))

#For a deeper understanding of gensim library, please go to - https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
"""
#NLTK Implementation - 

>>> from gensim.models import Word2Vec
>>> from nltk.corpus import brown, movie_reviews, treebank
>>> b = Word2Vec(brown.sents())
>>> mr = Word2Vec(movie_reviews.sents())
>>> t = Word2Vec(treebank.sents())
 
>>> b.most_similar('money', topn=5)
[('pay', 0.6832243204116821), ('ready', 0.6152011156082153), ('try', 0.5845392942428589), ('care', 0.5826011896133423), ('move', 0.5752171277999878)]
>>> mr.most_similar('money', topn=5)
[('unstoppable', 0.6900672316551208), ('pain', 0.6289106607437134), ('obtain', 0.62665855884552), ('jail', 0.6140228509902954), ('patients', 0.6089504957199097)]
>>> t.most_similar('money', topn=5)
[('short-term', 0.9459682106971741), ('-LCB-', 0.9449775218963623), ('rights', 0.9442864656448364), ('interested', 0.9430986642837524), ('national', 0.9396077990531921)]
 
>>> b.most_similar('great', topn=5)
[('new', 0.6999611854553223), ('experience', 0.6718623042106628), ('social', 0.6702290177345276), ('group', 0.6684836149215698), ('life', 0.6667487025260925)]
>>> mr.most_similar('great', topn=5)
[('wonderful', 0.7548679113388062), ('good', 0.6538234949111938), ('strong', 0.6523671746253967), ('phenomenal', 0.6296845078468323), ('fine', 0.5932096242904663)]
>>> t.most_similar('great', topn=5)
[('won', 0.9452997446060181), ('set', 0.9445616006851196), ('target', 0.9342271089553833), ('received', 0.9333916306495667), ('long', 0.9224691390991211)]
 
>>> b.most_similar('company', topn=5)
[('industry', 0.6164317727088928), ('technical', 0.6059585809707642), ('orthodontist', 0.5982754826545715), ('foamed', 0.5929019451141357), ('trail', 0.5763031840324402)]
>>> mr.most_similar('company', topn=5)
[('colony', 0.6689200401306152), ('temple', 0.6546304225921631), ('arrival', 0.6497283577919006), ('army', 0.6339291334152222), ('planet', 0.6184555292129517)]
>>> t.most_similar('company', topn=5)
[('panel', 0.7949466705322266), ('Herald', 0.7674347162246704), ('Analysts', 0.7463694214820862), ('amendment', 0.7282689809799194), ('Treasury', 0.719698429107666)]
"""