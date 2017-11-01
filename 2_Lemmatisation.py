"""
# Note, you can also use spacy in place of NLTK but it may pop error saying it requires Microsoft Visual C++ Build Tools
# In case you are using conda, use - 
#         python3 -m spacy.en.download --force all
#         (or)
#         conda install spacy
#         (or)
# In the above case, make sure you also type the following to create venv in above case. - conda create --name py36 python=3 This command 
# installs a few new packages: pip: 9.0.1-py36_1, python: 3.6.1-0, setuptools: 27.2.0-py36_1, vs2015_runtime: 14.0.25123-0
# ,wheel: 0.29.0-py36_0
# In case of venv -         
#        python -m venv spacy-venv
#        source spacy-venv/bin/activate
#        pip install -U pip
#        pip install -U spacy
#        python -m spacy download en         

import spacy
nlp=spacy.load("en")
doc="good better best"
for token in nlp(doc):
    print(token,token.lemma_)
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a")) #Default for pos is noun
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
print(lemmatizer.lemmatize("abaci"))
print(lemmatizer.lemmatize("is"))            # Gives 'is', since default pos is n
print(lemmatizer.lemmatize("are"))           # Gives 'are', since default pos is n
print(lemmatizer.lemmatize("is",pos="v"))    # Gives 'be'
print(lemmatizer.lemmatize("are",pos="v"))   # Gives 'be'


print("Without context")
print("Lemmatise %s: %s" % ("studying", lemmatizer.lemmatize("studying")))
print("Lemmatise %s: %s" % ("study", lemmatizer.lemmatize("study")))
print("Lemmatise %s: %s" % ("studies", lemmatizer.lemmatize("studies")))
print("Lemmatise %s: %s" % ("studied", lemmatizer.lemmatize("studied")))
#Without context
#Lemmatise studying: studying
#Lemmatise study: study
#Lemmatise studies: study
#Lemmatise studied: studied

print("With context")
print("Lemmatise %s: %s" % ("studying", lemmatizer.lemmatize("studying", pos="v")))
print("Lemmatise %s: %s" % ("study", lemmatizer.lemmatize("study", pos="v")))
print("Lemmatise %s: %s" % ("studies", lemmatizer.lemmatize("studies", pos="v")))
print("Lemmatise %s: %s" % ("studied", lemmatizer.lemmatize("studied", pos="v")))
#With context
#Lemmatise studying: study
#Lemmatise study: study
#Lemmatise studies: study
#Lemmatise studied: study