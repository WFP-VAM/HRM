import pytest
import sys
sys.path.append("../Src/")


COORDS_X = [12.407305, 6.864997, 12.407305, 32.544209, 18.572159, 11.493380, 12.407305, 6.864997, 12.407305, 32.544209]
COORDS_Y = [41.821816, 45.832565, 41.821816, 15.539874, 4.367064, 3.848012, 41.821816, 45.832565, 41.821816, 15.539874]
# -----------------------------------

#  TODO: earthengine-api cannot test on Travis cause authentication.
# def test_SentinelImages():
#     """ Testing the module that extracts the information from Sentinel Images. """
#     si = pytest.importorskip('sentinel_images')
#     simages = si.SentinelImages('../tests/')
#     simages.download(COORDS_X, COORDS_Y, start_date='2017-01-01', end_date='2018-01-01')
#     f = simages.featurize(COORDS_X, COORDS_Y, start_date='2017-01-01', end_date='2018-01-01')
#     assert f.shape == (len(COORDS_X), len(COORDS_Y))

# -----------------------------------s


def test_GoogleImages():
    """ Testing the module that extracts the information from Google Images. """
    gi = pytest.importorskip('google_images')
    gimages = gi.GoogleImages('../tests/')
    gimages.download(COORDS_X, COORDS_Y, step=False)
    f = gimages.featurize(COORDS_X, COORDS_Y, step=False)
    assert f.shape == (len(COORDS_X), len(COORDS_Y))

# -----------------------------------

#  TODO: earthengine-api cannot test on Travis cause authentication.
# def test_Nightlights():
#     """ Testing the module that extracts nightlight values. """
#     ni = pytest.importorskip('nightlights')
#     utils = pytest.importorskip('utils')
#
#     nlights = ni.Nightlights('../tests')
#     # get area of interest
#     area = utils.points_to_polygon(-9.348335, 10.349370, -9.254608, 10.413534)
#     nlights.download(area, '2016-01-01', '2017-01-01')
#     f = nlights.featurize([-9.3, -9.28], [10.37, 10.39])
#     assert (f[0]+f[1])/2 > 0.1


# -----------------------------------


def test_Acled():
    """ Testing the module that extracts ACLED data. """
    ad = pytest.importorskip('acled')
    coords_x, coords_y = [1.183056], [9.553300]
    acled = ad.ACLED("../tests")
    acled.download("TGO", '2017-01-01', '2018-01-01')
    d = {}
    for property in ["fatalities", "n_events", "violence_civ"]:
        for k in [10000, 100000]:
            d[property + "_" + str(k)] = acled.featurize(coords_x, coords_y, property=property, function='density', buffer=k)

    assert sum(d[item][0] for item in d) > 0