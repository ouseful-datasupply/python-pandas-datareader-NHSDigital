#Based on https://github.com/pydata/pandas-datareader/blob/master/pandas_datareader/wb.py

import warnings

import pandas as pd
import numpy as np

from pandas_datareader.base import _BaseReader
from pandas.compat import string_types

import requests
import lxml.html
import lxml.etree

import zipfile
#The following fudge copes with Python 2 and Python 3
try:
    from StringIO import StringIO as zreader
except ImportError:
    from io import BytesIO as zreader

import json

# This list of dataset codes was pulled from the NHS Digital Organisation
# Data Service data downloads pages:
# https://digital.nhs.uk/organisation-data-service/data-downloads

dataset_codes = ['epraccur', 'etrust', 'eccg', 'eccgsite', 'epcmem', 'epracmem',
                 'egdpprac', 'egpcur', 'egparc', 'epracarc', 'ehospice',
                 'epharmacyhq', 'edispensary', 'enurse', 'epcdp', 'eabeydispgp',
                 'ecarehomehq', 'ecarehomesite', 'ecarehomesucc', 'ephp',
                 'ephpsite', 'enonnhs', 'eprison', 'eschools', 'ejustice',
                 'ecare']
datatype_codes = ['gp-data', 'other-nhs', 'health-authorities',
                  'non-nhs', 'miscellaneous']

jdata=json.load(open("nhsdigitalods.json"))


class NHSDigitalOrganisationDataServiceReader(_BaseReader):
    """
    Download data series from NHS Digital's Organisation Data Service
    
    Parameters
    ----------
    datasets: NHS Digital ODS indicator string or list of strings
        taken from the filepath of the downloaded dataset.
    errors: str {'ignore', 'warn', 'raise'}, default 'warn'
        Country codes are validated against a hardcoded list.  This controls
        the outcome of that validation, and attempts to also apply
        to the results from world bank.
        errors='raise', will raise a ValueError on a bad country code.
        
    Returns
    -------
    ``pandas`` DataFrame.
    """
    
    def __init__(self, datasets=None, datatypes=None, errors='warn'):
        
        if datasets is None and datatypes is None:
            datatypes = datatype_codes
        
        if isinstance(datatypes, string_types):
            datatypes = [datatypes]
        if isinstance(datasets, string_types):
            datasets = [datasets]
        
        bad_datasets = np.setdiff1d(datasets, dataset_codes) if datasets is not None else []
        # Validate the input
        if len(bad_datasets) > 0:
            tmp = ", ".join(bad_datasets)
            if errors == 'raise':
                raise ValueError("Invalid dataset code(s): %s" % tmp)
            if errors == 'warn':
                warnings.warn('Non-standard dataset '
                              'codes: %s' % tmp, UserWarning)
                
        bad_datatypes = np.setdiff1d(datatypes, datatype_codes) if datatypes is not None else []
        # Validate the input
        if len(bad_datatypes) > 0:
            tmp = ", ".join(bad_datatypes)
            if errors == 'raise':
                raise ValueError("Invalid datatype code(s): %s" % tmp)
            if errors == 'warn':
                warnings.warn('Non-standard datatype '
                              'codes: %s' % tmp, UserWarning)

        self.datasets = datasets
        self.datatypes = datatypes
        self.errors = errors

    def get_datasets(self):
        """Download information about all NHS Digital Organisation Data Service datasets"""
        global _cached_dataset_lookups
        
        if isinstance(_cached_dataset_lookups, pd.DataFrame):
            return _cached_dataset_lookups.copy()
        
        lookupURLs=["https://digital.nhs.uk/organisation-data-service/data-downloads/gp-data",
             "https://digital.nhs.uk/organisation-data-service/data-downloads/other-nhs",
             "https://digital.nhs.uk/organisation-data-service/data-downloads/health-authorities",
             "https://digital.nhs.uk/organisation-data-service/data-downloads/non-nhs",
             "https://digital.nhs.uk/organisation-data-service/data-downloads/miscellaneous"]
           
        data = pd.DataFrame()

        for url in lookupURLs:
            txt=requests.get(url).text
            table = lxml.html.fromstring(txt)

            for row in table.xpath('//tr')[1:]:
                cells=row.xpath('td')
                if cells[1] is not None:
                    dataURL=cells[1].xpath('a/@href')[0]
                    data=pd.concat([data, pd.DataFrame([{'Label':cells[0].text,
                                                         'Date':cells[2].text,
                                                         'Period':cells[3].text,
                                                         'Dataset':dataURL.split('/')[-3],
                                                         'URL':dataURL,
                                                         'Type':url.split('/')[-1]}])])

        data = data.reset_index(drop=True)
        
        # cache
        _cached_dataset_lookups = data.copy()
        return data

    def search(self, string='GP', field='Dataset', case=False):
        """
        Search available datasets from NHS Digital Organisation Data Service
        Parameters
        ----------
        string: string
            regular expression
        field: string
            Label, Period, Dataset, Type, Date
            See notes below
        case: bool
            case sensitive search?
        Notes
        -----
        The first time this function is run it will download and cache the full
        list of available datasets. Depending on the speed of your network
        connection, this can take time. Subsequent searches will use the cached
        copy, so they should be much faster.
        Label : code label for dataset
        Period: reporting period
        Dataset: title of dataset
        Date: reporting date
        Type: general category dataset is listed under
        """
        datasets = self.get_datasets()
        data = datasets[field]
        idx = data.str.contains(string, case=case)
        out = datasets.loc[idx].dropna()
        return out

    def zipgrabber(self, url):
        ''' Grab a zip file from a URL '''
        r = requests.get(url)
        z = zipfile.ZipFile(zreader(r.content))
        return z

    def zipgrabberfile(self, url, f):
        ''' Grab a file by name from a zip file grabbed from a URL '''
        return self.zipgrabber(url).open(f)

    def zipfile(self, z, f):
        ''' Grab a file by name from an already grabbed zip file '''
        return z.open(f)

    def zipfilelist(self, z):
        ''' Return the names of files contained in a grabbed zip file '''
        return z.namelist()
    
    def read(self):
        return self._read()

    def _read(self):
        global _cached_datasets
        
        data = {}
        
        if self.datasets is None and self.datatypes is not None:
            datasets = []
            for dt in self.datatypes:
                datasets = datasets + _cached_dataset_lookups[_cached_dataset_lookups['Type']==dt]['Dataset'].tolist()
            self.datasets = [ds for ds in datasets if ds in dataset_codes ]
            
        for dataset in self.datasets:
            if dataset in _cached_datasets:
                data[dataset]= _cached_datasets[dataset]
                continue
            # Build URL for api call
            try:
                url=_cached_dataset_lookups[_cached_dataset_lookups['Dataset']==dataset].iloc[0]['URL']
                
                #Should trap as a warning
                if dataset not in jdata: return pd.DataFrame()
                
                names=jdata[dataset]['cols']
                dates=jdata[dataset]['dates']
                codes=jdata[dataset]['codes']
                index=jdata[dataset]['index']
                
                #Try to guess cols to cast as dates
                if dates=='auto':
                    dates=[names.index(c) for c in names if 'date' in c.lower()]

                #Try to guess columns that should be classed as string not int
                dtypes={c:str for c in names
                                        if "phone" in c.lower() or " code" in c.lower() or "type" in c.lower()}

                df = pd.read_csv(self.zipgrabberfile(url, '{}.csv'.format(dataset)),
                     header=None, names=None if names==[] else names, parse_dates=dates,
                     low_memory=False, encoding='Latin-1')
                
                for c in df.columns:
                    if c.startswith('Null') : df.drop(c, axis=1, inplace=True)
                if codes is not None:
                    for col in codes:
                        df[col + ' Value']=df[col].astype(str).map(codes[col])
                if index=='auto':
                    index=[names[0]]
                if index is not None: df=df.set_index(index)

                _cached_datasets[dataset]=df
                data[dataset]=df

            except ValueError as e:
                msg = str(e) + ' dataset: ' + dataset
                if self.errors == 'raise':
                    raise ValueError(msg)
                elif self.errors == 'warn':
                    warnings.warn(msg)

        # Confirm we actually got some data, and build Dataframe
        if len(data) > 1:
            return data
        elif len(data) == 1:
            return data[list(data)[0]]
        else:
            msg = "No datasets returned data."
            raise ValueError(msg)
            
            
def download(dataset=None, datatype=None,
             errors='warn', **kwargs):
    """
    Download datasets from NHS Digital Organisation Data Service
    
    Parameters
    ----------
    dataset: string or list of strings
        taken from the dataset codes in dataset URLs
    datatype: string or list of strings
        taken from the data collection page
    errors: str {'ignore', 'warn', 'raise'}, default 'warn'
        Dataset and datatype codes are validated against a hardcoded list.  This controls
        the outcome of that validation.
        errors='raise', will raise a ValueError on a bad dataset or datatype code.
    kwargs:
        keywords passed to NHSDigitalOrganisationDataServiceReader
    Returns
    -------
    ``pandas`` DataFrame with columns: country, iso_code, year,
    indicator value.
    """
    return NHSDigitalOrganisationDataServiceReader(datasets=dataset, datatypes=datatype,
                                                   **kwargs).read()

_cached_dataset_lookups = None
_cached_datasets = {}

def search(string='GP', field='Dataset', case=False, **kwargs):
    """
    Search available datasets from NHS Digital Organisation Data Service
    Parameters
    ----------
    string: string
        regular expression
    field: string
        Label, Period, Dataset, Type, Date
        See notes below
    case: bool
        case sensitive search?
    Notes
    -----
    The first time this function is run it will download and cache the full
    list of available datasets. Depending on the speed of your network
    connection, this can take time. Subsequent searches will use the cached
    copy, so they should be much faster.
    Label : code label for dataset
    Period: reporting period
    Dataset: title of dataset
    Date: reporting date
    Type: general category dataset is listed under
    """

    return NHSDigitalOrganisationDataServiceReader(**kwargs).search(string=string, field=field,
                                                                    case=case)