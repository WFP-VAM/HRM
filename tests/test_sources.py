import sys
sys.path.append("../Src/")

# -----------------------------------
from sentinel_images import SentinelImages

def test_SentinelImages():
    """ Testing the module that extracts the information from Sentinel Images. """
    simages = SentinelImages('../tmp/')
    simages.download([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')
    f = simages.featurize([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')
    assert f.shape == (2, 2)

# -----------------------------------
from google_images import GoogleImages

def test_GoogleImages():
    """ Testing the module that extracts the information from Google Images. """
    gimages = GoogleImages('../tmp/')
    gimages.download([12.407305, 6.864997], [41.821816, 45.832565], step=False)
    f = gimages.featurize([12.407305, 6.864997], [41.821816, 45.832565], step=False)
    assert f.shape == (2, 2)

# -----------------------------------
from nightlights import Nightlights
from utils import points_to_polygon

def test_Nightlights():
    """ Testing the module that extracts nightlight values. """
    nlights = Nightlights('../tmp')

    # get area of interest
    area = points_to_polygon(-9.348335, 10.349370, -9.254608, 10.413534)
    nlights.download(area, '2016-01-01', '2017-01-01')
    f = nlights.featurize([-9.3, -9.28], [10.37, 10.39])
    assert (f[0]+f[1])/2 > 0.1