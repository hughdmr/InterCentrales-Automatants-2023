# Training Data for S1 and S2 fusion
Google Cloud Storage Bucket: ```assignment-2023-zip```

The task is to generate an S2 cloudless image at time0 using S2 from time-1, time-2 and S1 from time0 and time-1 and time-2.
There is csv file in the dataset that creates a triplets of S2 images.

There is more files in the dataset, but use just the triplets created from `image_series.csv` file. 
Reasoning: other files are either clouded, corrupted, or doesn't have triplets.

There is 7635 triplets in the image. Please create your train/val/test split from them.

Triplets  e.g:
```
KALININGRAD_GUSEV_2018-04-07_2018-04-19-0-0-13-5.tiff,
KALININGRAD_GUSEV_2018-04-07_2018-04-19-1-0-13-5.tiff,
KALININGRAD_GUSEV_2018-04-07_2018-04-19-2-0-13-5.tiff
```
It's the same picture in time, where only difference is the measurement id. 
Where 2 is the most recent image and 0 is the oldest. Or in other words, 2 is time0, 1 is time-1 and 0 is time-2.

## Dataset structure

### Directory structure

* **s1/** data from [Sentinel-1]([Sentinel-1 - Missions - Sentinel Online - Sentinel Online](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1)) mission captured by [Synthetic Aperture Radar]([SAR Instrument - Sentinel-1 SAR Technical Guide - Sentinel Online - Sentinel Online](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/sar-instrument)). Data as two band tiffs: VV and VH polarisation. Values as dB in range [-30,0] 
* **s2/** data from [Sentinel-2]([Sentinel-2 - Missions - Sentinel Online - Sentinel Online](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)) mission captured by [MSI](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument). Values as [LAI index]([Leaf area index - Wikipedia](https://en.wikipedia.org/wiki/Leaf_area_index))
* **s2-mask/** Scene Classification Mask for S2. Mask values described [here](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm)

### File naming scheme

```<Country>_<Area>_<DateFrom>_<DateTo>-<Measurement_id>-<LSTM-#measurement-before>-<Row>-<Column>.tiff``` 
E.g. ```SPAIN_BADAJOZ_2019-03-11_2019-03-29-2-7-1-6.tiff```

The most important part is the ```<Measurement_id>``` which is the id of the measurement.
The ```<LSTM-#measurement-before>``` is in this case always zero, so no need to think about it.
Other parts of the naming are straightforward.

* Every s2 has s2 mask and vice versa.

### S1 & S2 pair
* All filenames mentioned in `image_series.csv` occur in `s1`,`s2` and `s2-mask` directories. 
* The difference between s1 and s2 dates is max +- 1 day

E.g
```
s1/KALININGRAD_GUSEV_2018-04-07_2018-04-19-0-0-13-5.tiff
s1/KALININGRAD_GUSEV_2018-04-07_2018-04-19-1-0-13-5.tiff
s1/KALININGRAD_GUSEV_2018-04-07_2018-04-19-2-0-13-5.tiff

s2/KALININGRAD_GUSEV_2018-04-07_2018-04-19-0-0-13-5.tiff
s2/KALININGRAD_GUSEV_2018-04-07_2018-04-19-1-0-13-5.tiff
s2/KALININGRAD_GUSEV_2018-04-07_2018-04-19-2-0-13-5.tiff

s2-mask/KALININGRAD_GUSEV_2018-04-07_2018-04-19-0-0-13-5.tiff
s2-mask/KALININGRAD_GUSEV_2018-04-07_2018-04-19-1-0-13-5.tiff
s2-mask/KALININGRAD_GUSEV_2018-04-07_2018-04-19-2-0-13-5.tiff
```

## Additional info  
E.g. `BULGARIA_SUMEN_2020-03-05_2020-03-17` means that s2 measurements were done on `2020-03-05` and `2020-03-17`. (the middle date is omitted)
And s1 measurements were done on `2020-03-06`, `2020-03-11`, `2020-03-16`.
The maximum difference between s1 and s2 dates for middle day is still +- 1 day. (in our case one of the options: `2020-03-10`, `2020-03-11`, `2020-03-12`)


```python
data = [
    {'name': 'BULGARIA_SUMEN_2020-03-05_2020-03-17', 'season': 'spring', 'dates':['2020-03-06', '2020-03-11', '2020-03-16'], 'bbox': [27.0770507038843853, 43.3107101995541228, 27.7273857047243801, 43.7546658274338256]},
    {'name': 'BULGARIA_SUMEN_2017-07-31_2017-08-12', 'season': 'summer', 'dates':['2017-08-01', '2017-08-06', '2017-08-11'], 'bbox': [27.0770507038843853, 43.3107101995541228, 27.7273857047243801, 43.7546658274338256]},
    {'name': 'BULGARIA_SUMEN_2018-08-13_2018-08-25', 'season': 'summer', 'dates':['2018-08-14', '2018-08-19', '2018-08-26'], 'bbox': [27.0770507038843853, 43.3107101995541228, 27.7273857047243801, 43.7546658274338256]},
    {'name': 'BULGARIA_SUMEN_2020-09-01_2020-09-13', 'season': 'fall', 'dates':['2020-09-02', '2020-09-07', '2020-09-14'], 'bbox': [27.0770507038843853, 43.3107101995541228, 27.7273857047243801, 43.7546658274338256]},
    {'name': 'BULGARIA_SUMEN_2019-12-10_2019-12-22', 'season': 'winter', 'dates':['2019-12-09', '2019-12-17', '2019-12-22'], 'bbox': [27.0770507038843853, 43.3107101995541228, 27.7273857047243801, 43.7546658274338256]},
    {'name': 'DENMARK_HOLSTEBRO_2018-07-06_2018-07-18', 'season': 'summer', 'dates':['2018-07-07', '2018-07-12', '2018-07-17'], 'bbox': [8.3165616668768951, 55.8892657046789836, 8.9667690266438314, 56.5766057827974436]},
    {'name': 'FRANCE_REIMS_2020-09-11_2020-09-23', 'season': 'fall', 'dates':['2020-09-12', '2020-09-17', '2020-09-22'], 'bbox': [4.1139977665082066, 49.2294376020799973, 4.7460411849720545, 49.7461080155861595]},
    {'name': 'GERMANY_ERFURTH_2020-03-31_2020-04-18', 'season': 'spring', 'dates':['2020-04-01', '2020-04-06', '2020-04-11'], 'bbox': [10.3302792937209986, 50.9748030700897630, 11.0576383035695489, 51.4326547053681082]},
    {'name': 'KALININGRAD_GUSEV_2018-04-07_2018-04-19', 'season': 'spring', 'dates':['2018-04-08', '2018-04-13', '2018-04-18'], 'bbox': [21.0428253538555126, 53.8334229490445537, 22.3173180832404441, 54.6143470471257118]},
    {'name': 'LATVIA_LITHUANIA_2019-04-17_2019-04-29', 'season': 'spring', 'dates':['2019-04-18', '2019-04-23', '2019-04-28'], 'bbox': [21.1675596294449022, 53.8351969955658731, 22.3307932026172757, 54.6176957719103555]},
    {'name': 'LATVIA_LITHUANIA_2019-05-29_2019-06-10', 'season': 'summer', 'dates':['2019-05-30', '2019-06-04', '2019-06-09'], 'bbox': [21.1675596294449022, 53.8351969955658731, 22.3307932026172757, 54.6176957719103555]},
    {'name': 'LATVIA_LITHUANIA_2020-09-18_2020-09-30', 'season': 'fall', 'dates':['2020-09-19', '2020-09-24', '2020-09-29'], 'bbox': [21.1675596294449022, 53.8351969955658731, 22.3307932026172757, 54.6176957719103555]},
    {'name': 'MOLDOVA_ROMANIA_2020-03-28_2020-04-09', 'season': 'spring', 'dates':['2020-03-29', '2020-04-03', '2020-04-08'], 'bbox': [26.0616418985187543, 47.9715609398838723, 26.8025262847258148, 48.4724998146488701]},
    {'name': 'MOLDOVA_ROMANIA_2019-08-20_2019-09-01', 'season': 'summer', 'dates':['2019-08-19', '2019-08-27', '2019-09-01'], 'bbox': [26.0616418985187543, 47.9715609398838723, 26.8025262847258148, 48.4724998146488701]},
    {'name': 'MOLDOVA_ROMANIA_2020-09-10_2020-09-22', 'season': 'fall', 'dates':['2020-09-10', '2020-09-15', '2020-09-22'], 'bbox': [26.0616418985187543, 47.9715609398838723, 26.8025262847258148, 48.4724998146488701]},
    {'name': 'SPAIN_BADAJOZ_2019-03-11_2019-03-29', 'season': 'spring', 'dates':['2019-03-11', '2019-03-16', '2019-03-23', '2019-03-28'], 'bbox': [-6.9351747872104452, 38.8941827844816785, -6.4522491566396845, 39.4490530935827621]},
    {'name': 'SPAIN_BADAJOZ_2017-07-13_2017-07-25', 'season': 'summer', 'dates':['2017-07-14', '2017-07-19', '2017-07-26'], 'bbox': [-6.9351747872104452, 38.8941827844816785, -6.4522491566396845, 39.4490530935827621]},
    {'name': 'SPAIN_BADAJOZ_2018-08-01_2018-08-25', 'season': 'summer', 'dates':['2018-07-31', '2018-08-08', '2018-08-13', '2018-08-20', '2018-08-25'], 'bbox': [-6.9351747872104452, 38.8941827844816785, -6.4522491566396845, 39.4490530935827621]},
    {'name': 'SPAIN_BADAJOZ_2018-10-18_2018-10-30', 'season': 'fall', 'dates':['2018-10-17', '2018-10-24', '2018-10-29'], 'bbox': [-6.9351747872104452, 38.8941827844816785, -6.4522491566396845, 39.4490530935827621]},
    {'name': 'SPAIN_BADAJOZ_2019-12-30_2020-01-11', 'season': 'winter', 'dates':['2019-12-31', '2020-01-05', '2020-01-12'], 'bbox': [-6.9351747872104452, 38.8941827844816785, -6.4522491566396845, 39.4490530935827621]},
    {'name': 'SWEDEN_MALMO_2020-04-10_2020-04-22', 'season': 'spring', 'dates':['2020-04-09', '2020-04-16', '2020-04-21'], 'bbox': [13.0922600942420324, 55.4901788977606500, 13.6597981364158318, 56.0012869238988671]},
    {'name': 'UKRAINE_KONOTOP_2020-03-12_2020-03-24', 'season': 'spring', 'dates':['2020-03-13', '2020-03-18', '2020-03-25'], 'bbox': [32.9298302777048804, 51.0479673319202547, 33.7536509646256988, 51.5523473443207578]},
    {'name': 'UKRAINE_KONOTOP_2020-07-15_2020-07-27', 'season': 'summer', 'dates':['2020-07-16', '2020-07-21', '2020-07-28'], 'bbox': [32.9298302777048804, 51.0479673319202547, 33.7536509646256988, 51.5523473443207578]},
    {'name': 'UKRAINE_KONOTOP_2020-09-01_2020-09-13', 'season': 'fall', 'dates':['2020-09-01', '2020-09-06', '2020-09-14'], 'bbox': [32.9298302777048804, 51.0479673319202547, 33.7536509646256988, 51.5523473443207578]},
    {'name': 'UK_CAMBRIDGE_2020-04-15_2020-04-27', 'season': 'spring', 'dates':['2020-04-16', '2020-04-21', '2020-04-26'], 'bbox': [0.1540780095497860, 51.7667251961464814, 0.8697022693650260, 52.1740050451253197]},
    {'name': 'UK_SALISBURY_2020-09-14_2020-09-26', 'season': 'fall', 'dates':['2020-09-13', '2020-09-21', '2020-09-26'], 'bbox': [-2.2365705476420641, 50.8989145337093944, -1.6045344136082365, 51.3958764144122000]},
    {'name': 'UK_WALES_2018-06-21_2018-07-03', 'season': 'summer', 'dates':['2018-06-22', '2018-06-27', '2018-07-02'], 'bbox': [-4.0238045739429955, 52.4136885948585558, -3.1889096041585487, 52.9405435364985593]},
]
```
