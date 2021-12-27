import cnn
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
Y = iris.target
Y_onehot = cnn.CNN.one_hot_encode(Y.reshape((-1,1)),num_cats=3)

print(X.shape,Y_onehot.shape)
# print(X,Y_onehot)

model = cnn.Model(optimiser_method='adam')

model.add_layer(
	cnn.layers.FC(3,input_shape=(4,1),activation='relu',initiation_method='kaiming_normal')
)
model.add_layer(
	cnn.layers.FC(3,activation='softmax',initiation_method='kaiming_normal')
)

model.prepare_model()
input()
# print(X[0:1],X[0:1].shape)
# print(Y[0:1],Y[0:1].shape, max(Y[0:1].shape))
model.train(X,Y_onehot,epochs=30,max_batch_size=64,cost_fn='cross_entropy',shuffle=True,learning_rate=0.1)
print(model.history)
# print(model.structure[-1].output)

plt.plot(model.history['cost'])
plt.show()