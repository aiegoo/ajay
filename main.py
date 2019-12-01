from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
#import matplotlib.image as mpimg
# define training data
import webbrowser
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.plot()
pyplot.savefig('testplot.png')
#img=mpimg.imread('testplot.png')
#imgplot = pyplot.imshow(img)
#pyplot.show()
webbrowser.open_new_tab('testplot.png')
#pyplot.show()
