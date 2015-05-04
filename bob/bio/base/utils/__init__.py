#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

from .resources import *
from .io import *
from .singleton import *

import numpy

def score_fusion_strategy(strategy_name = 'avarage'):
  """Returns a function to compute a fusion strategy between different scores.

  Different strategies are employed:

  * ``'average'`` : The averaged score is computed using the :py:func:`numpy.average` function.
  * ``'min'`` : The minimum score is computed using the :py:func:`min` function.
  * ``'max'`` : The maximum score is computed using the :py:func:`max` function.
  * ``'median'`` : The median score is computed using the :py:func:`numpy.median` function.
  * ``None`` is also accepted, in which case ``None`` is returned.
  """
  try:
    return {
        'average' : numpy.average,
        'min' : min,
        'max' : max,
        'median' : numpy.median,
        None : None
    }[strategy_name]
  except KeyError:
#    warn("score fusion strategy '%s' is unknown" % strategy_name)
    return None
