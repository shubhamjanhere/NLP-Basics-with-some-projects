# pip install sematch
# nltk.download('wordnet_ic')
# You also need to edit one of the sematch library files, sparql in case you are using python 3. You need to change the print statement.
from sematch.semantic.similarity import WordNetSimilarity
import pandas as pd

wns = WordNetSimilarity()

words = ['artist', 'musician', 'scientist', 'physicist', 'actor', 'movie']
sim_matrix = [[wns.word_similarity(w1, w2, 'wpath') for w1 in words] for w2 in words]
df = pd.DataFrame(sim_matrix, index=words,columns=words)
print(df)

print(wns.word_similarity("Dog", "Cat"))