import sys
sys.path.append("../Src/")
from sentinel_images import SentinelImages
from google_images import GoogleImages


def test_SentinelImages():
    """ Testing the module that extracts the information from Sentinel Images. """
    simages = SentinelImages('../tmp/')
    simages.download([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')
    f = simages.featurize([12.407305, 6.864997], [41.821816, 45.832565], start_date='2017-01-01', end_date='2018-01-01')
    assert f.shape == (2, 2)


def test_GoogleImages():
    """ Testing the module that extracts the information from Google Images. """
    gimages = GoogleImages('../tmp/')
    gimages.download([12.407305, 6.864997], [41.821816, 45.832565], step=False)
    f = gimages.featurize([12.407305, 6.864997], [41.821816, 45.832565], step=False)
    assert f.shape == (2, 2)