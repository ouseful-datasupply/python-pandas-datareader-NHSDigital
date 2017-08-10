from setuptools import setup

setup(name='pd_datareader_nhs',
      version='0.1',
      description='pandas data reader extensions for NHS Digital',
      url='https://github.com/psychemedia/python-pandas-datareader-NHSDigital',
      author='Tony Hirst',
      author_email='tony.hirst@gmail.com',
      license='MIT',
      packages=['pd_datareader_nhs'],
      install_requires=[
          'pandas',
          'numpy',
          'pandas_datareader',
          'requests',
          'lxml'
      ],
      zip_safe=False)