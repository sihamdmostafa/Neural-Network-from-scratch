from Modules.Loss1 import *
from Modules.Modules1 import *
import numpy as np
import matplotlib.pyplot as plt
from tools.tool import *
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

mnist = fetch_openml('mnist_784', version=1,data_home='files')
donnees = list(zip(np.array(mnist.data[:1000]), np.array(mnist.target[:1000], dtype = np.float64)))
np.random.shuffle(donnees)
data, y = zip(*donnees)
data=np.array(data)
y=np.array(y)

lin1 = Linear((data.shape[1],30),b = True)

in_size = data.shape[1]
h1_size = 100
h2_size = 10
value_training_test = [(range(int(len(data) * 0.9)), range(int(len(data) * 0.9), len(data)))]
for id_train, id_test in value_training_test:
    lin1 = Linear((in_size, 128),b = True)
    lin2 = Linear((128, in_size),b = True)
    
    #lin3 = Linear((h2_size, h1_size),b = True)
    #lin3.parameters = h2.parameters.T
    #lin4 = Linear(h1_size, in_size,b = True)
    #lin4.parameters = lin1.parameters.T
    
    Codeur   = [lin1, TanH(), lin2, Sigmoid()]
    #Decodeur = [lin3, Sigmoid(), lin4, Sigmoid()]
    
    seq = Sequentiel(m = Codeur)
    optim = Optim(seq, BCELoss(), 1e-3)
    
    moyenne,variance = SGD(data[id_train], data[id_train],optim,1,10)

plt.plot(moyenne)
plt.legend(('moyenne du loss'))
plt.show()

"""

X_tsne = TSNE(learning_rate=150.0).fit_transform(data)
plt.figure(figsize=(25,5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y)

X_tsne = TSNE(learning_rate=150.0).fit_transform(seq.predict(data))
plt.figure(figsize=(25,5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y)

"""

"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=1, max_iter=10).fit(seq.predict(data))
"""