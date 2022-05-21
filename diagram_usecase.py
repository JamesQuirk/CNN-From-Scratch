import cnn
import numpy as np
np.set_printoptions(linewidth=200)

X = np.random.normal(size=(3,12,12))

y = np.array([[0],[1]])


model = cnn.Model(input_shape=(3,12,12))
model.add_layer(
	cnn.layers.Conv2D(filt_shape=(3,3),num_filters=2,stride=1,padding=1)
)
model.add_layer(
	cnn.layers.Pool(filt_shape=(3,3),stride=3,pool_type='mean')
)
model.add_layer(
	cnn.layers.FC(num_nodes=9,activation='relu')
)
model.add_layer(
	cnn.layers.FC(num_nodes=2,activation='sigmoid')
)

model.train([X],[y],epochs=1)
