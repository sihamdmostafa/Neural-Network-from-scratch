from Modules.Loss1 import *
from Modules.Modules1 import *
import numpy as np
import matplotlib.pyplot as plt
from tools.tool import *
from sklearn.datasets import fetch_openml



mnist = fetch_openml('mnist_784', version=1,data_home='files')
donnees = list(zip(np.array(mnist.data[:1000]), np.array(mnist.target[:1000], dtype = np.float64)))
np.random.shuffle(donnees)
data, y = zip(*donnees)

score = []

seq = Sequentiel([Conv1D(3, 1, 32, stride=1),
                      MaxPool1D(2, 2),
                      Flatten(),
                      Linear((4064, 100),b=False),
                      ReLu(),
                      Linear((100, 10),b=False)
                      ])
optim = Optim(seq, CELoss(), 1e-3)        
    
moyenne,variance= SGD(data, y,optim, 1,10)
    
plt.plot(moyenne)
plt.legend(('moyenne du loss'))
plt.show()