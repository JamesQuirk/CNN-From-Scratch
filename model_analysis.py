from cnn.model import load_model
from model_analysis import CNN_Analyser

model = load_model('models/cnn_model_adam_tf_comparitor_vectorised_14-33-36.pkl')
CA = CNN_Analyser(model)

# print(model.get_model_details())
# for layer in model.structure:
# 	print(layer.output.mean())
# print(model.structure[-1].output)
CA.weight_distributions()
CA.display_filters(0)
CA.plot_cost_gradients()
CA.show_output_profiles()
CA.plot_cost()
# CA.output_distribution_trends()

CA.show()
