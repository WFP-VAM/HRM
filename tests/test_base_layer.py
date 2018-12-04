import sys
sys.path.append("../Src/")
from base_layer import BaseLayer
from utils import s3_download
import numpy as np


def test_base_layer():
    """ Testing the class that handles the raster base layer and the survey dataset. """
    s3_download('hrm-geofiles-rasters', 'Malawi_worldpop.tif', '../tests/tmp.tif')

    # base layer -12.898452, 33.344663
    GRID = BaseLayer('../tests/tmp.tif', [33.472016, 33.344663], [-13.051757, -12.898452])

    assert (np.round(GRID.x_size, 3) == 0.001) & (GRID.bands_data.min() == -999.0)