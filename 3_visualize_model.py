import gensim
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
from nltk.corpus import stopwords


# see also: https://www.quora.com/How-do-I-visualise-word2vec-word-vectors
# see also: https://raw.githubusercontent.com/devmount/GermanWordEmbeddings/master/visualize.py 
 
def main(pca=True):

    wv, vocabulary = load_embeddings("hamlet.vec")

    if pca:
        pca = PCA(n_components=2, whiten=True)
        Y = pca.fit(wv[:300,:]).transform(wv[:300,:])
    else:
        tsne = TSNE(n_components=2, random_state=0)
        Y = tsne.fit_transform(wv[:500,:])

    np.set_printoptions(suppress=True)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label.lower() not in stopwords.words('english'):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
 
def load_embeddings(file_name):

 
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in if len(line.strip().split()) != 2]) 

        wv = np.loadtxt(wv)

    return wv, vocabulary


 
if __name__ == '__main__':
    main(pca=False)
