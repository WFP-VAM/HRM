import sys
sys.path.append("../Src/")
import pytest


def test_base_layer():
    """ Testing the class that handles the raster base layer and the survey dataset. """
    base_layer = pytest.importorskip('base_layer')
    utils = pytest.importorskip('utils')
    np = pytest.importorskip('numpy')
    utils.s3_download('hrm-geofiles-rasters', 'Malawi_worldpop.tif', '../tests/tmp.tif')

    # base layer -12.898452, 33.344663
    GRID = base_layer.BaseLayer('../tests/tmp.tif', [33.472016, 33.344663], [-13.051757, -12.898452])

    assert (np.round(GRID.x_size, 3) == 0.001) & (GRID.bands_data.min() == -999.0)