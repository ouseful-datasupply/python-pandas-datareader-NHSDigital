# python-pandas-datareader-NHSDigital
Pandas data reader extensions to access NHS Digital organisation data service datasets

Data scraped weekly by a Github Action and published using `datasette` to https://nhs-ods.psychemedia.now.sh/

Example db: 
https://lite.datasette.io/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fouseful-datasupply%2Fpython-pandas-datareader-NHSDigital%2Fmaster%2Fnhs_ods.db%3Fraw%3Dtrue#/

Example IW postcode glob query: https://lite.datasette.io/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fouseful-datasupply%2Fpython-pandas-datareader-NHSDigital%2Fmaster%2Fnhs_ods.db%3Fraw%3Dtrue#/nhs_ods/edispensary?_filter_column_1=Postcode&_filter_op_1=glob&_filter_value_1=PO3%5B0-9%5D*&_filter_column=&_filter_op=exact&_filter_value=&_sort=rowid

## Usage

PyPi: `pip install ouseful_nhs_datasupply`

Install using: `pip3 install --force-reinstall --upgrade --no-deps git+https://github.com/ouseful-datasupply/python-pandas-datareader-NHSDigital.git`

See worked example notebook: https://github.com/ouseful-datasupply/python-pandas-datareader-NHSDigital/blob/master/Testing%20-%20python-pandas-datareader-NHSDigital.ipynb

Import using:

`import ouseful_nhs_datasupply.nhs_digital_ods as ods`

This gives access to generic tools on dymanically creater datatreader instances.

Create an object from the datareader class as:

`import ouseful_nhs_datasupply.nhs_digital_ods import NHSDigitalOrganisationDataServiceReader`

and then:

`ods_reader = NHSDigitalOrganisationDataServiceReader()`

Search for available datasets - this scrapes pages linked from the [NHS Digital Organisation Data Service *Data downloads* page](https://digital.nhs.uk/organisation-data-service/data-downloads) to build up a catalogue of available datasets.

*Note - not all datasets are necessarily available - the codes are filtered through a whitelist in the `nhs_digital_ods.py` file.*

```python
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

```python
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

## Command Line Interface

A CLI to the organisation data service download that downloads all ODS reference tables into a single SQLIte3 database.

See the database running as a [datasette](https://github.com/simonw/datasette) at: [https://ousefulnhsdata.herokuapp.com/](https://ousefulnhsdata.herokuapp.com/)

```text
Usage: nhs_admin [OPTIONS] COMMAND

Commands: 
  collect        Fresh collection of all NHS Organisation Data Service files       

Options:
  --dbname TEXT  SQLite database name (default: nhs_ods.db)
  --help         Show this message and exit.
```

For example, to download everything: `nhs_admin collect`
