from Modules.Loss1 import *
from Modules.Modules1 import *
import numpy as np
import matplotlib.pyplot as plt
from tools.tool import *
from sklearn.datasets import fetch_openml


def f(x):
    return np.argmax(x, axis = 1)


mnist = fetch_openml('mnist_784', version=1,data_home='files')
donnees = list(zip(np.array(mnist.data[:1000]), np.array(mnist.target[:1000], dtype = np.float64)))
np.random.shuffle(donnees)
data, y = zip(*donnees)
data=np.array(data)
y=np.array(y)
lin1 = Linear((data.shape[1],30),b = True)
lin2 = Linear((30,20),b = True)
sequentiel = Sequentiel(m = [lin1, TanH(), lin2,Softmax()], fsortie = f)
optim = Optim(sequentiel, CELoss(), 1e-3)        
    
moyenne,variance= SGD(data,y,optim,1,10)
    
plt.plot(moyenne)
plt.legend(('moyenne du loss'))
plt.show()

#print(sequentiel.accuracy(data,y))