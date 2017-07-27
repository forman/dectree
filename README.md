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
variable possible states, e.g. one of {HIGH, MIDDLE, LOW} or one of {FAST, SLOW}. Each property in the fuzzy set
is represented by a mapping function that transforms the variable's value (a floating point number of known range)
into a fuzzy truth value  (in the range of zero to one) which verifies the property. Logical expressions that
combine these truth values translate as follows with *a* and *b* being fuzzy truth values:

* *a* and *b* --> min(*a*, *b*)
* *a* or *b* --> max(*a*, *b*)
* not *a* --> 1 - *a*

For example, the condition `x is HIGH and y is not SLOW` translates to `min(HIGH(x), 1 - SLOW(y))`.

Available mapping functions are

* `ramp_down(x, x1=0.0, x2=0.5)` - a ramp with negative slope for x in the range x1 to x2, 1 if x < x1,
  and 0 if x > x2.
* `ramp_up(x, x1=0.5, x2=1.0)` a ramp with positive slope for x in the range x1 to x2, 0 if x < x1,
  and 1 if x > x2.
* `triangle(x, x1=0.0, x2=0.5, x3=1.0)` - a ramp with positive slope for x in the range x1 to x2,
  a ramp with negative slope in the range x2 to x3, 0 if x < x1 or x >  x3.
* `true(x)` - always 1.
* `false(x)` - always 0.

The property mapping functions are defined in [dectree/propfuncs.py](https://github.com/forman/dectree/blob/master/dectree/propfuncs.py).

## Limitations

Output values must have types whose property mapping functions can only be `true()` or `false()`.

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
    
