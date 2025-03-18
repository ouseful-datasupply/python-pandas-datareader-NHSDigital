from setuptools import setup

setup(
    name="ouseful_nhs_datasupply",
    version="0.3.1",
    description="pandas data reader extensions for NHS Digital",
    url="https://github.com/psychemedia/python-pandas-datareader-NHSDigital",
    author="Tony Hirst",
    author_email="tony.hirst@gmail.com",
    license="MIT",
    packages=["ouseful_nhs_datasupply"],
    package_data={"ouseful_nhs_datasupply": ["data/*.json"]},
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "requests_cache",
        "lxml",
    ],
    entry_points="""
        [console_scripts]
        nhs_admin=ouseful_nhs_datasupply.cli:cli
    """,
    zip_safe=False,
)
