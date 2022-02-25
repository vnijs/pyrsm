# PYRSM

Python functions and classes for Customer Analytics at the Rady School of Management (RSM), University of California, San Diego (UCSD)

## Install instructions

If you have python and pip3 install you can use the below to install the latest version of `pyrsm`:

`pip3 install --user 'pyrsm>=0.5.3'`

or 

`conda install 'pyrsm>=0.5.3'`

## Example notebooks

### Data

* [Load example datasets](examples/load-example-data.ipynb)
* [Save and load notebook state files](examples/save-load-state.ipynb)

### Basics

* [Cross-tabs](examples/basics-cross-tabs.ipynb)
* [Correlation](examples/basics-correlation.ipynb)

## Radiant

The examples above are (mostly) connected to the example data and analyses initially created for the family of Radiant R packages. See  https://radiant-rstats.github.io/docs/index.html for more details

&copy; Vincent Nijs (2022) <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank"><img alt="Creative Commons License" style="border-width: 0" src="images/by-nc-sa.png"/></a>

## Links

* [Pandas cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
* [Pandas vs R](https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html) 
* [Pandas vs R](https://cheatsheets.quantecon.org/stats-cheatsheet.html)
* [Pandas vs R](https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07)
* [Split-Appy-Combine in Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)
* [Split-Appy-Combine in Pandas](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/)
* ["assign" in Pandas](http://queirozf.com/entries/mutate-for-pandas-dataframes-examples-with-assign)
* [Condition count](https://stackoverflow.com/a/45752640/1974918)
* [`value_counts` in Pandas](https://appdividend.com/2019/01/24/pandas-series-value_counts-tutorial-with-example/)
* Convert column to string or categorical:
    - `df["zipcode"] = df.zipcode.astype(str)`
    - `df["zipcode"] = df.zipcode.astype('category')`

## Python Learning Resources

* Python for everyone (https://www.py4e.com)
* Scientific computing with Python (https://www.freecodecamp.org/learn/scientific-computing-with-python/)
* Data analysis with Python (https://www.freecodecamp.org/learn/data-analysis-with-python/)

## Python AND R

* [Reticulate](https://rstudio.github.io/reticulate/articles/calling_python.html)
* [rpy2](https://rpy2.github.io/doc/v3.3.x/html/notebooks.html)

<!-- 
## Statistics

* [Variance Inflation Factor](https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python) 
-->