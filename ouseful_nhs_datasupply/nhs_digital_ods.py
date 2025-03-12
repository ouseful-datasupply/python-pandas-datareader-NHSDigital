# Based on https://github.com/pydata/pandas-datareader/blob/master/pandas_datareader/wb.py

import warnings

import sqlite3

import pandas as pd
import numpy as np


# from pandas.compat import string_types
from pandas import read_sql_query, DataFrame, read_csv

import requests
import requests_cache

import lxml.html
import lxml.etree

import zipfile

# The following fudge copes with Python 2 and Python 3
try:
    from StringIO import StringIO as zreader
except ImportError:
    from io import BytesIO as zreader

import json

DEFAULT_DB = "nhs_ods.sqlite"

# Need unique column names
# https://stackoverflow.com/a/2837551/454773
def rename_duplicates(old):
    seen = {}
    for x in old:
        if x in seen:
            seen[x] += 1
            yield "%s_%d" % (x, seen[x])
        else:
            seen[x] = 0
            yield x


# This list of dataset codes was pulled from the NHS Digital Organisation
# Data Service data downloads pages:
# https://digital.nhs.uk/organisation-data-service/data-downloads

dataset_codes = [
    "epraccur",
    "etrust",
    "eccg",
    "eccgsite",
    "epcmem",
    "epracmem",
    "egdpprac",
    "egpcur",
    "egparc",
    "epracarc",
    "ehospice",
    "epharmacyhq",
    "edispensary",
    "enurse",
    "epcdp",
    "eabeydispgp",
    "ecarehomehq",
    "ecarehomesite",
    "ecarehomesucc",
    "ephp",
    "ephpsite",
    "enonnhs",
    "eschools",
    "ejustice",
    "ecare",
]
datatype_codes = [
    "gp-and-gp-practice-related-data",
    "other-nhs-organisations",
    "health-authorities-and-support-agencies",
    "non-nhs-organisations",
    "miscellaneous",
]

import pkg_resources

JSON_FILE = pkg_resources.resource_filename(
    "ouseful_nhs_datasupply", "data/nhsdigitalods.json"
)
jdata = json.load(open(JSON_FILE))


class NHSDigitalOrganisationDataServiceReader:
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

    def __init__(
        self,
        datasets=None,
        datatypes=None,
        sqlite3db=True,
        errors="warn",
        use_cache=True,
        cache_name=".nhs_ods_cache",
        cache_expire_after=3600,
        cache_backend="sqlite",
    ):

        self.session = None

        # Set up the session with or without caching
        if use_cache:
            import requests_cache

            self.session = requests_cache.CachedSession(
                cache_name=cache_name,
                backend=cache_backend,
                expire_after=cache_expire_after,
            )
            print(f"Caching enabled with {cache_backend} backend: {cache_name}")
        else:
            self.session = requests.Session()
            print("Caching disabled")

        if datasets is None and datatypes is None:
            datatypes = datatype_codes

        if isinstance(datatypes, str):
            datatypes = [datatypes]
        if isinstance(datasets, str):
            datasets = [datasets]

        bad_datasets = (
            np.setdiff1d(datasets, dataset_codes) if datasets is not None else []
        )
        # Validate the input
        if len(bad_datasets) > 0:
            tmp = ", ".join(bad_datasets)
            if errors == "raise":
                raise ValueError("Invalid dataset code(s): %s" % tmp)
            if errors == "warn":
                warnings.warn("Non-standard dataset " "codes: %s" % tmp, UserWarning)

        bad_datatypes = (
            np.setdiff1d(datatypes, datatype_codes) if datatypes is not None else []
        )
        # Validate the input
        if len(bad_datatypes) > 0:
            tmp = ", ".join(bad_datatypes)
            if errors == "raise":
                raise ValueError("Invalid datatype code(s): %s" % tmp)
            if errors == "warn":
                warnings.warn("Non-standard datatype " "codes: %s" % tmp, UserWarning)

        self.datasets = datasets
        self.datatypes = datatypes
        self.errors = errors

        self._cached_datasets = {}
        self._cached_dataset_lookups = None

        if isinstance(sqlite3db, bool):
            if sqlite3db:  # If db is True
                self.dbname = DEFAULT_DB
            else:  # If db is False
                self.dbname = None
        elif sqlite3db is None:
            self.dbname = None
        else:
            # db is string or path
            self.dbname = sqlite3db
        self.sqlite3con = None
        if sqlite3db:
            self.sqlite3con = sqlite3.connect(self.dbname)
            self._setdb()

        self._sourceDatasets(False)

    def init(self):
        pass

    def _setdb(self):
        if self.sqlite3con and not self._dbtable_exists("dataset_date"):
            print("Setting up a new dataset_date table...")
            DataFrame({"Dataset": dataset_codes, "Date": None}).to_sql(
                con=self.sqlite3con, name="dataset_date", index=False
            )
            if isinstance(self._cached_dataset_lookups, DataFrame):
                self._updatedb("_cached_dataset_lookups", self._cached_dataset_lookups)
            for key in self._cached_datasets:
                self._updatedb(key, self._cached_datasets[key])

    def _sourceDatasets(self, retval=True):
        # print("Sourcing datasets")
        lookupURLs = [
            "https://digital.nhs.uk/services/organisation-data-service/data-downloads/gp-and-gp-practice-related-data",
            "https://digital.nhs.uk/services/organisation-data-service/data-downloads/other-nhs-organisations",
            "https://digital.nhs.uk/services/organisation-data-service/data-downloads/health-authorities-and-support-agencies",
            "https://digital.nhs.uk/services/organisation-data-service/data-downloads/non-nhs-organisations",
            "https://digital.nhs.uk/services/organisation-data-service/data-downloads/miscellaneous",
        ]

        data = DataFrame()

        for url in lookupURLs:
            # print(f'Looking up {url}...')
            # TO DO - should really handle exception better here, eg if there is no connection?
            try:
                txt = self.session.get(url).text
                table = lxml.html.fromstring(txt)

                for row in table.xpath("//table/tbody/tr"):
                    cells = row.xpath("td")
                    if cells[1] is not None:
                        dataURL = cells[1].xpath("a/@href")[0]
                        # File contents	File	Date uploaded	Quarterly/Monthly	Available in XML	Role code	API call
                        data = pd.concat(
                            [
                                data,
                                DataFrame(
                                    [
                                        {
                                            "Label": cells[0].text,
                                            "Date": cells[2].text,
                                            "Period": cells[3].text,
                                            "Dataset": dataURL.split("/")[-1].split(
                                                "."
                                            )[0],
                                            "URL": dataURL,
                                            "Type": url.split("/")[-1],
                                        }
                                    ]
                                ),
                            ]
                        )
                data = data.reset_index(drop=True)
            except:
                print(f"There was an issue somewhere parsing {url}")
                if self._dbtable_exists("_cached_dataset_lookups"):
                    data = self.read_db(table="_cached_dataset_lookups")
                else:
                    warnings.warn("Couldn't scrape ODS data listing: {}".format(url))

        # cache
        self._cached_dataset_lookups = data.copy()
        if self.sqlite3con and not self._dbtable_exists("_cached_dataset_lookups"):
            self._updatedb("_cached_dataset_lookups", self._cached_dataset_lookups)
        if not retval:
            return
        return data

    def get_datasets(self):
        """Download information about all NHS Digital Organisation Data Service datasets"""

        if isinstance(self._cached_dataset_lookups, DataFrame):
            return self._cached_dataset_lookups.copy()
        return self._sourceDatasets()

    def search(self, string="GP", field="Dataset", case=False):
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
        """Grab a zip file from a URL"""
        r = self.session.get(url)
        z = zipfile.ZipFile(zreader(r.content))
        return z

    def zipgrabberfile(self, url, f):
        """Grab a file by name from a zip file grabbed from a URL"""
        return self.zipgrabber(url).open(f)

    def zipfile(self, z, f):
        """Grab a file by name from an already grabbed zip file"""
        return z.open(f)

    def zipfilelist(self, z):
        """Return the names of files contained in a grabbed zip file"""
        return z.namelist()

    def read_db(self, q=None, table=None):
        return self._read_db(q, table)

    def _read_db(self, q=None, table=None):
        # Note that the db can be stale - should check against recently loaded _cached_dataset_lookups?
        if not self.sqlite3con:
            return DataFrame()
        if q is None and table is None:
            q = "SELECT name FROM sqlite_master WHERE type='table'"
        elif q is None:
            q = "SELECT * FROM {tbl}".format(tbl=table)
        return read_sql_query(q, self.sqlite3con)

    def _checkdbcopyiscurrent(self, table):
        if table != "_cached_dataset_lookups" and self._dbtable_exists(table):
            datadate = self._cached_dataset_lookups[
                self._cached_dataset_lookups["Dataset"] == table
            ].iloc[0]["Date"]
            q = "SELECT Date FROM dataset_date WHERE Dataset='{dataset}';".format(
                dataset=table
            )
            dbdatadate = self.read_db(q).iloc[0]["Date"]
            return datadate == dbdatadate
        return False

    def _updatedb(self, table=None, data=None):
        if not self.sqlite3con or not isinstance(data, DataFrame) or table is None:
            # print("Not cacheing data in db...")
            return
        con = self.sqlite3con
        #print("Cacheing data in db")
        data.to_sql(con=con, name=table, if_exists="replace", index=False)
        if table != "_cached_dataset_lookups" and table != "dataset_date":
            datadate = self._cached_dataset_lookups[
                self._cached_dataset_lookups["Dataset"] == table
            ].iloc[0]["Date"]
            q = "UPDATE dataset_date SET Date='{date}' WHERE Dataset='{dataset}';".format(
                date=datadate, dataset=table
            )
            c = con.cursor()
            c.execute(q)
            con.commit()

    def _dbtable_exists(self, table=None):
        if not self.sqlite3con or table is None:
            return False
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name='{}'".format(
            table
        )
        return True if len(self.read_db(q)) else False

    def cached(self):
        tmp = self.search(string="")
        return tmp[tmp["Dataset"].isin(self._cached_datasets.keys())]

    def read(self):
        return self._read()

    def _read(self):

        data = {}

        # Get a list of datasets for a particular datatype
        if self.datasets is None and self.datatypes is not None:
            datasets = []
            for dt in self.datatypes:
                datasets = (
                    datasets
                    + self._cached_dataset_lookups[
                        self._cached_dataset_lookups["Type"] == dt
                    ]["Dataset"].tolist()
                )
            self.datasets = [ds for ds in datasets if ds in dataset_codes]

        for dataset in self.datasets:
            # Use a pre-existing copy of the dataset if we have one
            if dataset in self._cached_datasets:
                data[dataset] = self._cached_datasets[dataset]
                continue
            elif self._dbtable_exists(dataset):
                # if we have a recent copy, use it
                if self._checkdbcopyiscurrent(dataset):
                    data[dataset] = self.read_db(table=dataset)
                    continue

            # Build URL for API call
            try:
                url = self._cached_dataset_lookups[
                    self._cached_dataset_lookups["Dataset"] == dataset
                ].iloc[0]["URL"]

                # Should trap as a warning
                if dataset not in jdata:
                    return DataFrame()

                names = list(rename_duplicates(jdata[dataset]["cols"]))
                dates = jdata[dataset]["dates"]
                codes = jdata[dataset]["codes"]
                index = jdata[dataset]["index"]

                # Try to guess cols to cast as dates
                if dates == "auto":
                    dates = [names.index(c) for c in names if "date" in c.lower()]

                # Try to guess columns that should be classed as string not int
                dtypes = {
                    c: str
                    for c in names
                    if "phone" in c.lower()
                    or " code" in c.lower()
                    or "type" in c.lower()
                }

                df = read_csv(
                    self.zipgrabberfile(url, "{}.csv".format(dataset)),
                    header=None,
                    names=None if names == [] else names,
                    parse_dates=dates,
                    low_memory=False,
                    encoding="Latin-1",
                )

                for c in df.columns:
                    if c.startswith("Null"):
                        df.drop(c, axis=1, inplace=True)
                if codes is not None:
                    for col in codes:
                        df[col + " Value"] = df[col].astype(str).map(codes[col])
                # The db table writer ignores the index...
                # if index=='auto':
                #    index=[names[0]]
                # if index is not None: df=df.set_index(index)

                if dataset not in self._cached_datasets:
                    self._cached_datasets[dataset] = df
                self._updatedb(dataset, df)
                data[dataset] = df

            except ValueError as e:
                msg = str(e) + " dataset: " + dataset
                if self.errors == "raise":
                    raise ValueError(msg)
                elif self.errors == "warn":
                    warnings.warn(msg)

        # Confirm we actually got some data, and build Dataframe
        if len(data) > 1:
            return data
        elif len(data) == 1:
            return data[list(data)[0]]
        else:
            msg = "No datasets returned data."
            raise ValueError(msg)


def download(dataset=None, datatype=None, errors="warn", sqlite3db=None, **kwargs):
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
    return NHSDigitalOrganisationDataServiceReader(
        datasets=dataset, datatypes=datatype, sqlite3db=sqlite3db, **kwargs
    ).read()


def search(string="GP", field="Dataset", case=False, **kwargs):
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

    return NHSDigitalOrganisationDataServiceReader(**kwargs).search(
        string=string, field=field, case=case
    )


def setdb(sqlite3db=DEFAULT_DB , **kwargs):
    """
    Enable a db.
    """
    NHSDigitalOrganisationDataServiceReader(sqlite3db=sqlite3db, **kwargs)._setdb()


def updatedb(sqlite3db=DEFAULT_DB , tables="all", **kwargs):
    """
    Populate the SQLite3 database. This may take some time.
    """
    datasets = None
    datatypes = None
    if isinstance(tables, str):
        if tables == "all":
            datasets = dataset_codes
        elif tables in dataset_codes:
            datasets = [tables]
        elif tables in datatype_codes:
            datatypes = [tables]
    elif isinstance(tables, list):
        for table in tables:
            if table in dataset_codes:
                datasets.append(table)
            elif table in datatype_codes:
                datatypes.append(table)
    download(dataset=datasets, datatype=datatypes, sqlite3db=sqlite3db, **kwargs)


def availableDatasets(typ="offline", **kwargs):
    if typ == "offline":
        return NHSDigitalOrganisationDataServiceReader().read_db(
            q='SELECT * FROM dataset_date WHERE Date!="None"'
        )
    return NHSDigitalOrganisationDataServiceReader().cached()


def init(sqlite3db=None, **kwargs):
    NHSDigitalOrganisationDataServiceReader(sqlite3db=sqlite3db, **kwargs).init()
