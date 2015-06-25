# A singleton class decorator, based on http://stackoverflow.com/a/7346105/3301902

class Singleton:
  """
  A non-thread-safe helper class to ease implementing singletons.
  This should be used as a **decorator** -- not a metaclass -- to the class that should be a singleton.

  The decorated class can define one `__init__` function that takes an arbitrary list of parameters.

  To get the singleton instance, use the :py:meth:`instance` method. Trying to use :py:meth:`__call__` will result in a :py:class:`TypeError` being raised.

  Limitations:

  * The decorated class cannot be inherited from.
  * The documentation of the decorated class is replaced with the documentation of this class.
  """

  def __init__(self, decorated):
    self._decorated = decorated
    # see: functools.WRAPPER_ASSIGNMENTS:
    self.__doc__ = decorated.__doc__
    self.__name__ = decorated.__name__
    self.__module__ = decorated.__module__
    self.__bases__ = []

    self._instance = None

  def create(self, *args, **kwargs):
    """Creates the singleton instance, by passing the given parameters to the class' constructor."""
    self._instance = self._decorated(*args, **kwargs)

  def instance(self):
      """Returns the singleton instance.
      The function :py:meth:`create` must have been called before."""
      if self._instance is None:
        raise RuntimeError("The class has not yet been instantiated using the 'create' method.")
      return self._instance

  def __call__(self):
    raise TypeError('Singletons must be accessed through the `instance()` method.')

  def __instancecheck__(self, inst):
    return isinstance(inst, self._decorated)
