from src.cnn import CNN
import mnist_dataloader
# np.set_printoptions(linewidth=200)


train_images, train_labels, test_images, test_labels = mnist_dataloader.get_data()

model = CNN(optimiser_method='gd')

# model.add_layer(
# 	CNN.Conv_Layer(filt_shape=(5,5),num_filters=5,stride=2,pad_type='include')
# )
# model.add_layer(
# 	CNN.Pool_Layer(filt_shape=(3,3),stride=1,pool_type='max',pad_type='include')
# )
model.add_layer(
	CNN.Flatten_Layer(input_shape=(28,28))
)
model.add_layer(
	CNN.FC_Layer(128,activation='relu',initiation_method=None)
)
model.add_layer(
	CNN.FC_Layer(128,activation='relu',initiation_method=None)
)
model.add_layer(
	CNN.FC_Layer(10,activation='softmax',initiation_method=None)
)
model.prepare_model()

input('Run training?')

train_finish, training_duration = model.train(
	train_images[:500],
	train_labels[:500],
	epochs=15,
	max_batch_size=128,
	shuffle=False,
	cost_fn='cross_entropy',
	learning_rate=0.1
)
print(f'Training duration: {training_duration}')
save_name = f'nn_model_sgd_{train_finish.strftime("%H-%M-%S")}.pkl'
print(f'Model saved to: {save_name}')
model.save_model(save_name)
