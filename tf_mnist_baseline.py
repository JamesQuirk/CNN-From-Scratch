'''
TensorFlow baseline model -- 86.7% test accuracy
'''


import tensorflow as tf
import mnist_dataloader

train_images, train_labels, test_images, test_labels = mnist_dataloader.get_data(normalise=False)

model = tf.keras.Sequential()

model.add(
	tf.keras.layers.Flatten(input_shape=(28,28))
)
model.add(
	tf.keras.layers.Dense(128,activation='relu')
)
model.add(
	tf.keras.layers.Dense(128,activation='relu')
)
model.add(
	tf.keras.layers.Dense(10,activation='softmax')
)

model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['accuracy']
)

history = model.fit(
	x=train_images,
	y=train_labels,
	epochs=3,
	batch_size=5000,
	shuffle=False
)

print(history.history)

results = model.evaluate(x=test_images,y=test_labels)

print(results)
