from src.cnn import CNN
from src.cnn_analyser import CNN_Analyser

model = CNN.load_model('nn_model_sgd_19-50-16.pkl')
CA = CNN_Analyser(model)

# print(model.get_model_details())
# for layer in model.structure:
# 	print(layer.output.mean())
# print(model.structure[-1].output)
# CA.weight_distributions()
# CA.display_filters(0)
CA.plot_cost_gradients()
CA.show_output_profiles()
CA.plot_cost()
# CA.output_distribution_trends()

CA.show()
