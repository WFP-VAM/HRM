# HRM

High Resolution Mapping of Food Security

The first step consists in replicating the Science paper *Combining satellite imagery and machine learning to predict poverty* in Uganda with LSMS 2011 data. 

Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., & Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. Science, 353(6301), 790-794.

This is done through the following Notebooks:

  1. Extracting Indicators from LSMS 2011-12 Uganda
  2. Download Images from Google Static API
  3. Generating features from fine-tuned CNN on nightlights (caffe).ipynb
  4. Predicting HH indicators with CNN features.ipynb
  
 The code is following closely the one shared on Neal Jean's Guthub repo: https://github.com/nealjean/predicting-poverty
  
The data structure to run the code is:
  
 ```
 
Data
├── Datasets
│   ├── Raw
│   └── processses_survey.csv
├── Network
├── Outputs
└── Satellite
    ├── raster_of_the_world.tif
    ├── Bing
    │   ├── 25161_9138
    │   │   └── 25161_9138.jpg
    └── Google
        ├── 8866_8866
        │   └── 8866_8866.jpg
        ├── 8869_8869
        │   └── 8869_8869.jpg
        └── 9169_9169
            └── 9169_9169.jpg
  ```
  
  processses_survey.csv should contain at least 3 columns: "gpsLongitude","gpsLatitude" and one indicator. You can either work with individual survey data or aggregate the surveys to some geographic level. 
  
  
 The next steps are (work in progress):
+ Applying the same methods in WFP assessments
  + Uganda Karamoja 2016 Assessment to begin with
  + Try Food Security indicators and Wealth indicators
+ Trying with different sources of imageries
  + Use less images per cluster (as for WFP assessments we know precisely the location of clusters)
  + Landsat/Sentinel data : lower resolution but more frequent and consistency of collection period 
  + Other sources of very high resolution imageries
+ Trying with different Neural Networks 
  + Comparing scores With/Whithout the transfer learning step
  + Trying with Networks available on Keras
  + Extract features from previous layers in the network
+ Adding features coming from other sources 
  + Ex: conflict GIS maps or CDR data (WorldPop approach) 

