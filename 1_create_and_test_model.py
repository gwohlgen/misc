import gensim
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

def train_model(fileid):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(gutenberg.sents(fileid), min_count=5, size=300, workers=4, window=10, sg=1, negative=5, iter=5)


if __name__ == '__main__':

    model = train_model('milton-paradise.txt')
    print(model.most_similar(positive=['God']))
    print(model.most_similar(positive=['Satan', 'God'], negative=['king']))


    model = train_model('shakespeare-hamlet.txt')
    print(model.most_similar(positive=['Hamlet']))
    print(model.most_similar(positive=['King']))
    print(model.most_similar(positive=['great']))


    ## save model
    model.save("hamlet.model") # binary format
    model.wv.save_word2vec_format("hamlet.vec", binary=False) # text / vec format
