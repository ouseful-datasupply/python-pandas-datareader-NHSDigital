# python-pandas-datareader-NHSDigital
Pandas data reader extensions to access NHS Digital datasets


## Usage

Install using: `pip3 install --force-reinstall --upgrade git+https://github.com/psychemedia/python-pandas-datareader-NHSDigital.git`

Import using:
```
import pd_datareader_nhs.nhs_digital_ods as ods
```

Search for available datasets - this scrapes pages linked from the [NHS Digital Organisation Data Service *Data downloads* page](https://digital.nhs.uk/organisation-data-service/data-downloads) to build up a catalogue of available datasets.

*Note - not all datasets are necessarily available - the codes are filtered through a whitelist in the `nhs_digital_ods.py` file.*

```
#Return all Labels
ods.search(string='', field='Label')

#Search for Label containing GP
ods.search(string='GP', field='Label', case=True)

#Search for datasets with a code containing 'cur'
ods.search(string='cur', field='Dataset')

#Search for datasets released as 'gp-data'
ods.search(string='gp-data', field='Type')

ss=ods.search('')
ss['Type'].unique()
>>> array(['gp-data', 'other-nhs', 'health-authorities', 'non-nhs',
       'miscellaneous'], dtype=object)
```

Actual datasets may be downloaded as *pandas* dataframes either one at a time or as a dict containing several different dataframes.

```    
dd=ods.download(datatype='other-nhs')
dd.keys()
>>> dict_keys(['eccg', 'eccgsite', 'etrust', 'ecare'])
#dd[key] returns a dataframe

dd=ods.download('ecarehomesucc')
>>> dd is a dataframe

dd=ods.download(['ecarehomesucc','epraccur'])
>>> dd is a dict containing one or more dataframes
dd['epraccur']
>>> dd['epraccur'] is a dataframe
```

The downloader should be cacheing data as it is downloaded, which can add a considerable overhead. At the moment, the downloaded data is *not* persisted using local storage. 