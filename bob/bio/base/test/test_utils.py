import numpy

import bob.bio.base
import bob.io.base.test_utils


def test_resources():
    # get list of extensions
    extensions = bob.bio.base.extensions()
    assert isinstance(extensions, list)
    assert "bob.bio.base" in extensions


def test_sampling():
    # test selection of elements
    indices = bob.bio.base.selected_indices(100, 10)
    assert indices == list(range(5, 100, 10))

    indices = bob.bio.base.selected_indices(100, 300)
    assert indices == range(100)

    indices = bob.bio.base.selected_indices(100, None)
    assert indices == range(100)

    array = numpy.arange(100)
    elements = bob.bio.base.selected_elements(array, 10)
    assert (elements - numpy.arange(5, 100, 10) == 0.0).all()

    elements = bob.bio.base.selected_elements(array, 200)
    assert (elements - numpy.arange(100) == 0.0).all()

    elements = bob.bio.base.selected_elements(array, None)
    assert (elements - numpy.arange(100) == 0.0).all()
