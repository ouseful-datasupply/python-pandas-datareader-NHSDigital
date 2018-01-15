from setuptools import setup

setup(name='pd_datareader_nhs',
      version='0.2.1',
      description='pandas data reader extensions for NHS Digital',
      url='https://github.com/psychemedia/python-pandas-datareader-NHSDigital',
      author='Tony Hirst',
      author_email='tony.hirst@gmail.com',
      license='MIT',
      packages=['pd_datareader_nhs'],
      package_data={'pd_datareader_nhs': ['data/*.json']},
      install_requires=[
          'pandas',
          'numpy',
          'pandas_datareader',
          'requests',
          'lxml'
      ],
      entry_points='''
        [console_scripts]
        nhs_admin=pd_datareader_nhs.cli:cli
    ''',
      zip_safe=False)