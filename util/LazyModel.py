import logging
log = logging.getLogger("de.t_animal.MA.util.LazyModel")

"""
	This is a wrapper around gensim model which only instantiates the wrapped
	model if (and only if) it is subscripted ([]) or a property is accessed
	(useful in conjunction with caches).
	After instantiating it, this class's object "morphs" into an object of
	the wrapped class (i.e. it casts itself and replaces its datastructures).
"""

class LazyModel():

	def __init__(self, modelConstructor, *args, **kwargs):
		"""
			@param modelConstructor - this method will be called to instantiate the model (usually a constructor)
			@param *args, **kwargs - all further arguments and keyword arguments will be passed to modelConstructor
		"""

		self.modelConstructor = modelConstructor
		self.args = args
		self.kwargs = kwargs

	def _instantiate(self):
		log.debug("Instantiating model")
		model = self.modelConstructor(*self.args, **self.kwargs)

		log.debug("Morphing into model object")
		del self.modelConstructor
		del self.args
		del self.kwargs

		self.__class__ = model.__class__
		self.__dict__ = model.__dict__

	def __getattr__(self, attr):
		self._instantiate()
		return getattr(self, attr)

	def __getitem__(self, key):
		self._instantiate()
		return self[key]