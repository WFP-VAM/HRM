import pandas as pd

def scoring_postprocess(features):
    # postprocess
    features = features.transpose().reset_index()
    features["i"] = features["index"].str.slice(0, 5)
    features["j"] = features["index"].str.slice(6, 10)
    features["i"] = pd.to_numeric(features["i"])
    features["j"] = pd.to_numeric(features["j"])

    return features