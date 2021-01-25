from src.cnn import CNN
import numpy as np
np.set_printoptions(linewidth=200)

X = np.random.normal(size=(3,12,12))

y = np.array([[0],[1]])


model = CNN(input_shape=(3,12,12))
model.add_layer(
	CNN.Conv_Layer(filt_shape=(3,3),num_filters=2,stride=1,padding=1)
)
model.add_layer(
	CNN.Pool_Layer(filt_shape=(3,3),stride=3,pool_type='mean')
)
model.add_layer(
	CNN.FC_Layer(num_nodes=9,activation='relu')
)
model.add_layer(
	CNN.FC_Layer(num_nodes=2,activation='sigmoid')
)

model.train([X],[y],epochs=1)
