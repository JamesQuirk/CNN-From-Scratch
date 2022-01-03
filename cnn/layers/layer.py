import numpy as np
import sys
from cnn.params import CNNParam
from typing import Tuple, Union

class Layer:
	'''
	ABSTRACT LAYER CLASS FOR ALL LAYER TYPES
	'''
	def __init__(self):
		self.model = None

		self.next_layer = None
		self.prev_layer = None

		self.output = None

	def _initiate_history(self):
		out_init_arr = np.zeros(self.model.EPOCHS * self.model.N)
		out_init_arr[:] = np.NaN
		cg_init_arr = np.zeros(self.model.TOTAL_ITERATIONS)
		cg_init_arr[:] = np.NaN
		self.history = {
			'output':
				{'mean':out_init_arr,'std':out_init_arr,'max':out_init_arr,'min':out_init_arr,'sum':out_init_arr,'median':out_init_arr},
			'cost_gradient':
				{'mean':cg_init_arr,'std':cg_init_arr,'max':cg_init_arr,'min':cg_init_arr,'sum':cg_init_arr,'median':cg_init_arr}
		}

	def _track_metrics(self,output=None,cost_gradient=None):
		if output is not None:
			self.history['output']['mean'][self.model.feed_forwards_cycle_index] = np.mean(output)
			self.history['output']['std'][self.model.feed_forwards_cycle_index] = np.std(output)
			self.history['output']['max'][self.model.feed_forwards_cycle_index] = np.max(output)
			self.history['output']['min'][self.model.feed_forwards_cycle_index] = np.min(output)
			self.history['output']['sum'][self.model.feed_forwards_cycle_index] = np.sum(output)
			self.history['output']['median'][self.model.feed_forwards_cycle_index] = np.median(output)
		if cost_gradient is not None:
			if not np.isnan(self.history['cost_gradient']['mean'][self.model.iteration_index]):
				print(f"Warning: value already set. Overwriting {self.history['cost_gradient']['mean'][self.model.iteration_index]} with {np.mean(cost_gradient)}")
				sys.exit()	# TODO: change
			self.history['cost_gradient']['mean'][self.model.iteration_index] = np.mean(cost_gradient)
			self.history['cost_gradient']['std'][self.model.iteration_index] = np.std(cost_gradient)
			self.history['cost_gradient']['max'][self.model.iteration_index] = np.max(cost_gradient)
			self.history['cost_gradient']['min'][self.model.iteration_index] = np.min(cost_gradient)
			self.history['cost_gradient']['sum'][self.model.iteration_index] = np.sum(cost_gradient)
			self.history['cost_gradient']['median'][self.model.iteration_index] = np.median(cost_gradient)

	def define_details(self):
		details = {
			'LAYER_INDEX':self.MODEL_STRUCTURE_INDEX,
			'LAYER_TYPE':self.LAYER_TYPE
		}
		if self.LAYER_TYPE is 'CONV':
			details.update({
				'NUM_FILTERS':self.NUM_FILTERS,
				'STRIDE':self.STRIDE
			})
		elif self.LAYER_TYPE is 'POOL':
			details.update({
				'STRIDE':self.STRIDE,
				'POOL_TYPE':self.POOL_TYPE
			})
		elif self.LAYER_TYPE is 'FLATTEN':
			details.update({
			})
		elif self.LAYER_TYPE is 'FC':
			details.update({
				'NUM_NODES':self.NUM_NODES,
				'ACTIVATION':self.ACTIVATION
			})
		elif self.LAYER_TYPE is 'ACTIVATION':
			details.update({
				'FUNCTION':self.FUNCTION
			})
		
		return details

	def count_params(self,split_trainable=True) -> Union[Tuple, int]:
		""" Sums sizes of any parameter attributes of the layer object.
		'parameter' is defined as any attribute that is of type 'CNNParam'.

		Returns: Tuple(trainable, non trainable) [if split_trainable is True]; total params otherwise.
		"""
		trainable = 0
		non_trainable = 0
		for param in self.get_params():
			if param.trainable:
				trainable += param.size
			else:
				non_trainable += param.size
		if split_trainable:
			return trainable, non_trainable
		else:
			return trainable + non_trainable

	def get_params(self):
		params = []
		for att in self.__dict__.values():
			if isinstance(att,CNNParam):
				params.append(att)
		return params

	@property
	def trainable(self):
		try:
			return self._trainable
		except AttributeError as e:
			# Defaults to 'True'
			self.trainable = True
			return self._trainable 

	@trainable.setter
	def trainable(self,value):
		""" When setting to trainability of the layer, this should link with the param trainability; 
		i.e. set the value for each param.
		This relationship is one-directional.
		"""
		assert isinstance(value,bool), f"{self}.trainable must be a boolean value."
		self._trainable = value
		for param in self.get_params():
			param.trainable = value
