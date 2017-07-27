# dectree
Decision Tree utilizing Fuzzy Logic

## Purpose

The *dectree* tool reads the definition of a decision tree and outputs a Python function that implements it.
The generated Python code is highly optimized through the use of [Numba](https://numba.pydata.org/). The 
function can be generated for inputs/outputs whose values are either scalars or vectors, usually given as 
[Numpy-like](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html) arrays.

The decision tree definition comprises the description inputs, outputs, and a set of rules.
The rules are if/else-statements whose conditional expressions compare one or more input variables with 
the individual properties of an associated *fuzzy set*. The if/else-bodies can again contain other rules
or output assignments. An output variable is assigned one of the properties of its associated fuzzy set.

Therefore the fuzzy set can be seen as a variable's data type defined in terms of the possible
variable states, e.g. one of {HIGH, MIDDLE, LOW} or one of {FAST, SLOW}. Each property in the fuzzy set
is a mapping from the variable's value (a floating point number of known range) to a fuzzy truth value 
(in the range of zero to one) which verifies the property. Logical expressions that combine these truth 
values translate as follows with *a* and *b* being fuzzy truth values:

* *a* and *b* --> min(*a*, *b*)
* *a* or *b* --> max(*a*, *b*)
* not *a* --> 1 - *a*

For example, the condition `x is HIGH and y is not SLOW` translates to `min(HIGH(x), 1 - SLOW(y))`.   

## Usage

    $ dectree -h
    $ dectree examples/im_classif.yml -o . --vectorize 
    
See also related notebook
[examples/im_classif.ipynb](https://github.com/forman/dectree/blob/master/examples/im_classif.ipynb).

## Installation

### Requirements

Command-line tool:

* Python 3.3+
* pyyaml

To run the generated Python modules and to run the *dectree* unit-tests:

* numba
* numpy


### Checkout
    
    $ git clone https://github.com/forman/dectree.git
    
### Install

    $ cd dectree
    $ python setup.py develop
    
