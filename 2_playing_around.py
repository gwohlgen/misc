import gensim
import gensim.downloader as api

from gensim.models import FastText
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

if __name__ == '__main__':
    """
        nice article: https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/
    """

    # load model
    model = gensim.models.KeyedVectors.load_word2vec_format("hamlet.vec", binary=False) # text / vec format


    ## what is the similarity between these words
    print(model.similarity('great', 'well'))
    print(model.similarity('great', 'death'))
    print(model.similarity('Hamlet', 'well'))
    print(model.similarity('King', 'Queene'))

    ## most similar to a single word
    print("King", model.most_similar("King"))

    ## words that are most far away from input .. 
    ## nothing useful there for this model
    print(model.most_similar(negative=['Lord', "Reynol"]))
    print(model.most_similar(negative=['Hamlet']))


    # model.most_similar(positive=['like','teh'],negative=['the'])
    print("king-queen", model.most_similar(positive=["woman",'King'],negative=['man']) )
    #print("superlative: good-better", model.most_similar(positive=["good",'bigger'],negative=['big']) )
    print("superlative: play-plays", model.most_similar(positive=["play",'goes'],negative=['go']) )

    text = api.load('text8')
    model = FastText(text, size=4, window=3, min_count=5, iter=10)
    print("king-queen", model.most_similar(positive=["woman",'King'],negative=['man']) )
    #print("superlative: good-better", model.most_similar(positive=["good",'bigger'],negative=['big']) )
    print("superlative: play-plays", model.most_similar(positive=["play",'goes'],negative=['go']) )


