import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from numpy import dtype
# nltk.download('punkt')

stemmer = PorterStemmer()
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype= np.float32)

    for idx , w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# sentence = ['hello', 'hi', 'hey']
# words = ['hey', 'are', 'you','doing', 'hello', 'hi', 'thats', 'cool']
# bag = bag_of_word(sentence, words)
# print(bag)