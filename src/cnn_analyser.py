import matplotlib.pyplot as plt
import numpy as np
import math

class CNN_Analyser:
	def __init__(self,model):
		self.model = model

	def display_filters(self,layer_index,delay_show=True):
		Conv_Layer = self.model.structure[layer_index]
		assert Conv_Layer.LAYER_TYPE == 'CONV'
		n_filts = Conv_Layer.filters.shape[0]	# rows
		n_channels = Conv_Layer.filters.shape[1]	# cols
		fig, axes = plt.subplots(nrows=n_filts,ncols=n_channels,constrained_layout=True)
		fig.suptitle(f'Filters; Layer: {layer_index}')
		if axes.ndim == 1:
			axes = axes.reshape((axes.shape[0],1))

		for f_ind in range(n_filts):
			for ch_ind in range(n_channels):
				im = axes[f_ind,ch_ind].imshow(Conv_Layer.filters[f_ind,ch_ind])
				axes[f_ind,ch_ind].figure.colorbar(im,ax=axes[f_ind,ch_ind])
		# fig.tight_layout()
		if not delay_show: plt.show()

	def weight_distributions(self,delay_show=True):
		# INCLUDES ONLY CONV AND FC LAYERS
		included_layers = [l for l in self.model.structure if l.LAYER_TYPE in ('CONV','FC')]
		n_layers = len(included_layers)
		fig, axes = plt.subplots(nrows=n_layers,ncols=1,constrained_layout=True)
		fig.suptitle('Weight Distributions')
		for i, layer in enumerate(included_layers):
			if layer.LAYER_TYPE == 'CONV':
				d = layer.filters.flatten()
			elif layer.LAYER_TYPE == 'FC':
				d = layer.weights.flatten()
			frqs, bins, _ = axes[i].hist(x=d,bins='auto')
			axes[i].set_title(f'Layer type: {layer.LAYER_TYPE} | model index: {layer.MODEL_STRUCTURE_INDEX}')
			axes[i].text(bins.max(),frqs.max(),f'mean: {d.mean():.2f}\nstd: {d.std():.2f}',horizontalalignment='center',verticalalignment='top')
		if not delay_show: plt.show()

	def output_distribution_trends(self,delay_show=True):
		''' 
		For each layer, shows how output distribution has changed over the itterations
		- top row plot will be means
		- bottom row plot will be stds
		'''
		n_layers = self.model.layer_counts['total']
		fig, axes = plt.subplots(nrows=2,constrained_layout=True)
		fig.suptitle('Output Distribution Trends')
		axes = axes.flatten()
		axes[0].set_title(f'Output Means')
		axes[0].set_xlabel('Feed forwards cycle')
		axes[1].set_title(f'Output Standard Deviations')
		axes[1].set_xlabel('Feed forwards cycle')
		for layer in self.model.structure:
			axes[0].plot(layer.history['output']['mean'],label=f'ID: {layer.MODEL_STRUCTURE_INDEX} | Type: {layer.LAYER_TYPE}')
			axes[1].plot(layer.history['output']['std'],label=f'ID: {layer.MODEL_STRUCTURE_INDEX} | Type: {layer.LAYER_TYPE}')
		axes[0].legend()
		axes[1].legend()
		if not delay_show: plt.show()

	def plot_cost_gradients(self,delay_show=True):
		n_layers = self.model.layer_counts['total']
		if n_layers < 5:
			n_cols = 1
			n_rows = n_layers
		elif n_layers < 10:
			n_cols = 2
			n_rows = math.ceil(n_layers / n_cols)
		else:
			n_cols = 3
			n_rows = math.ceil(n_layers / n_cols)
		fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,constrained_layout=True)
		fig.suptitle('Cost Gradients')
		for layer_ind in range(n_layers):
			ax = axes.flatten()[layer_ind]
			ax.plot(self.model.structure[layer_ind].history['cost_gradient']['median'])
			ax.set_title(f'layer: {layer_ind}',fontsize=8)
			ax.set_xlabel('Iteration',fontsize=8)
			ax.set_ylabel('Median Cost Gradient',fontsize=8)

		# fig.tight_layout()
		if not delay_show: plt.show()

	def plot_cost(self,delay_show=True):
		fig, ax = plt.subplots(constrained_layout=True)
		fig.suptitle('Cost')
		ax.plot(self.model.history['cost'])
		if not delay_show: plt.show()

	def show_output_profiles(self,delay_show=True):
		fig,ax = plt.subplots(constrained_layout=True)
		fig.suptitle('Summary of outputs through the model')
		ax.set_xlabel('Layer index')
		xs = np.arange(self.model.layer_counts['total'])
		for func, lab in [(np.mean,'mean'),(np.max,'max'),(np.min,'min'),(np.std,'std')]:
			ys = np.array([func(layer.output) for layer in self.model.structure])
			ax.plot(xs,ys,label=lab)
		ax.legend()
		for x in xs:
			act = self.model.structure[x].LAYER_TYPE
			ax.text(x,0,act,fontsize=8,horizontalalignment='center')
		if not delay_show: plt.show()

	def show_weight_profiles(self,delay_show=True):	# TODO: not finished
		fig,ax = plt.subplots(constrained_layout=True)
		fig.suptitle('Summary of outputs through the model')
		ax.set_xlabel('Layer index')
		xs = np.arange(self.model.layer_counts['total'])
		for func, lab in [(np.mean,'mean'),(np.max,'max'),(np.min,'min'),(np.std,'std')]:
			ys = np.array([func(layer.weights) for layer in self.model.structure])
			ax.plot(xs,ys,label=lab)
			ax.legend()
		for x in xs:
			act = self.model.structure[x].LAYER_TYPE
			ax.text(x,0,act,fontsize=8,horizontalalignment='center')
		if not delay_show: plt.show()

	@staticmethod
	def display_img(img,delay_show=True):
		fig, ax = plt.subplots(constrained_layout=True)
		ax.imshow(img)
		if not delay_show: plt.show()

	@staticmethod
	def show():
		plt.show()

