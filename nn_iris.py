from src import cnn
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
Y = iris.target
Y_onehot = cnn.CNN.one_hot_encode(Y.reshape((-1,1)),num_cats=3)

print(X.shape,Y_onehot.shape)
model = cnn.CNN(input_shape=(1,1,4),optimiser_method='gd')

model.add_layer(
	cnn.CNN.FC_Layer(3,activation='relu',initiation_method=None)
)
model.add_layer(
	cnn.CNN.FC_Layer(3,activation='softmax',initiation_method=None)
)

model.prepare_model()
# print(X[0:1],X[0:1].shape)
# print(Y[0:1],Y[0:1].shape, max(Y[0:1].shape))
model.train(X,Y_onehot,epochs=3,max_batch_size=32,cost_fn='cross_entropy')
print(model.history)
# print(model.structure[-1].output)

plt.plot(model.history['cost'])
plt.show()