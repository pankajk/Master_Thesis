import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

#use all digits
#mnist = fetch_mldata("MNIST original")
mnist = fetch_mldata("MNIST Original")
mytargets = list(range(0,10))
#mytargets = [3,8]
XX_train, yy_train = mnist.data / 255., mnist.target
X_train=[]
y_train=[]
for i, label in enumerate(yy_train):
  if label in mytargets:
    X_train.append(XX_train[i])
    y_train.append(yy_train[i])
num_samples_to_plot = 5000
X_train, y_train = shuffle(X_train, y_train)
X_train, y_train = X_train[:num_samples_to_plot], y_train[:num_samples_to_plot]  # lets subsample a bit for a first impression

for digit in mytargets:
  instances=[i for i in y_train if i==digit]
  print ("Digit",digit,"appears ",len(instances), "times")

pca = PCA(n_components=2)
fig, plot = plt.subplots()
fig.set_size_inches(50, 50)
plt.prism()

X_transformed = pca.fit_transform(X_train)
plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)
plot.set_xticks(())
plot.set_yticks(())

count=0;
plt.tight_layout()
plt.suptitle("PCA for MNIST digits ")
for label , x, y in zip(y_train, X_transformed[:, 0], X_transformed[:, 1]):
#Lets annotate every 1 out of 200 samples, otherwise graph will be cluttered with anotations
  if count % 200 == 0:
    plt.annotate(str(int(label)),xy=(x,y), color='black', weight='normal',size=10,bbox=dict(boxstyle="round4,pad=.5", fc="0.8"))
  count = count + 1
#plt.savefig("mnist_pca.png")
plt.show()
