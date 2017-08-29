# dectree
Decision Tree utilizing Fuzzy Logic

## Purpose

The *dectree* tool reads the definition of a decision tree and outputs a Python function that implements it.
The generated Python code is highly optimized through the use of [Numba](https://numba.pydata.org/). The 
function can be generated for inputs/outputs whose values are either scalars or vectors, usually given as 
[Numpy-like](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html) arrays.

The decision tree definition comprises the description inputs, outputs, and a set of rules.
The rules are if/else-statements whose conditional expressions compare one or more input variables with 
the individual properties defined for a given variable type. In the fuzzy logic literature,
each property defines a *fuzzy set* and an associated *membership function* is used to determine the 
membership (truth value in the range 0 to 1) of given value of a *linguistic variable* with respect to that property. 
The if/else-bodies can again contain other rules or contain output variable assignments. 
Again, only properties can be assigned to output variables. In contrast to the properties for input variables,
output properties must deliver constant truth values in the range 0 to 1.

The type for a variable `x` may comprise the properties `HIGH`, `MIDDLE`, `LOW` while the type for `y`
may define `FAST`, and `SLOW`.
 
A property's membership function maps a variable value (a floating point number of known range) 
to a fuzzy truth value (in the range of zero to one) which is a measure for the membership of the value to that 
property. Logical expressions that combine these truth values translate as follows with 
*a* and *b* being fuzzy truth values:

* *a* and *b* --> min(*a*, *b*)
* *a* or *b* --> max(*a*, *b*)
* not *a* --> 1 - *a*

For example, the condition `x is HIGH and y is not SLOW` "fuzzifies" to `min(HIGH(x), 1 - SLOW(y))`.

Available membership functions are

* `ramp_down(x1=0.0, x2=0.5)` - a ramp with negative slope for x in the range x1 to x2, 1 if x < x1,
  and 0 if x > x2.
* `ramp_up(x1=0.5, x2=1.0)` a ramp with positive slope for x in the range x1 to x2, 0 if x < x1,
  and 1 if x > x2.
* `triangle(x1=0.0, x2=0.5, x3=1.0)` - a ramp with positive slope for x in the range x1 to x2,
  a ramp with negative slope in the range x2 to x3, 0 if x < x1 or x > x3.
* `true()` - always 1;
* `false()` - always 0;
* `const(t)` - always `t`.

The membership functions are defined in [dectree/propfuncs.py](https://github.com/forman/dectree/blob/master/dectree/propfuncs.py).

## Limitations

The only membership functions allowed for properties assigned to output values are `true()`, `false()`, or `const(t)`.

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
    
    
## How it works
    
### Syntax


A rule has the general form

    if <CONDITION>:
        <BODY>
        
or        
        
    if <CONDITION>:
        <BODY_1>
    else:
        <BODY_2>

where `<BODY>` may be another nested rule or comprise a list of one or more output variable 
assignments of the form
        
        <OUTPUT> = <PROPERTY>
            
where `<OUTPUT>`is the name of any defined output and `<PROPERTY>` is the name of a property
defined for the output type. The value of output properties Currently, the only membership functions supported for outputs 
are the ones that do not depend on the output value: `true()`, `false()`, and `const(t)`.

The final value of an output variable in the if-body `<BODY_1>` of a rule is computed by the minimum 
of the current truth value given by `<CONDITION>` and the constant value returned by the membership function
of the assigned output property.  

Likewise, the final value of an output variable in the else-body `<BODY_2>` of a rule is computed 
by the minimum of the negation of current truth value given by `<CONDITION>` and the 
constant value returned by the membership function of the assigned output property.  

The rule's `<CONDITION>` is a conditional expression comprising comparisons of the form 
`<INPUT> is <PROPERTY>` which can be combined using the logical `and`, `or`, 
and `not` operators having the common precedences. Parentheses can be used to control
expression precedences. A conditional expression `<CONDITION>` is translated by a function
`translate()` as follows:

* `<INPUT> is <PROPERTY>` or also `<INPUT> == <PROPERTY>` translates into a function call `<TYPE>_<PROPERTY>(<INPUT>)` that computes the 
   truth value of `<INPUT>` with respect to the given property `<PROPERTY>` defined for type `<TYPE>`.
* `not <CONDITION>` translates into `1.0 - translate(<CONDITION>)` 
* `<CONDITION_1> and <CONDITION_2>` translates into `min(translate(<CONDITION_1>), translate(<CONDITION_2>))` 
* `<CONDITION_1> or <CONDITION_2>` translates into `max(translate(<CONDITION_1>, translate(<CONDITION_2>))`
 
### Translation

A simple rule of the form

    if <CONDITION>:
        <OUTPUT_1> = <VALUE_1>
    else:
        <OUTPUT_2> = <VALUE_2>
        
will translate into 
        
    t0 = 1.0
    
    # if <CONDITION>:
    t1 = min(t0, translate(<CONDITION>))

    #     <OUTPUT_1> = <VALUE_1>
    <OUTPUT_1> = min(t1, <VALUE_1>)
    
    # else:
    t1 = min(t0, 1.0 - t1)

    #     <OUTPUT_2> = <VALUE_2>
    <OUTPUT_2> = min(t1, <VALUE_2>)
              
The following example has nested rules and the output `<OUTPUT_2>` is assigned twice. 

    if <COND_1>:
        if <COND_2>:
            if <COND_3>:
                <OUTPUT_1> = <VALUE_1>
                <OUTPUT_2> = <VALUE_2>
    else:
        if <COND_4>:
            <OUTPUT_3> = <VALUE_3>
        else:
            if <COND_5>:
                <OUTPUT_4> = <VALUE_4>
            else:
                <OUTPUT_2> = <VALUE_2>
        
Multiple assignments to the same variable are interpreted as alternatives,
thus the maximum value of all possible values for `<OUTPUT_2>` is taken 
(see last line). The translation is 

    t0 = 1.0

    # if <COND_1>:
    t1 = min(t0, translate(<COND_1>))

    #     if <COND_2>:
    t2 = min(t1, translate(<COND_2>))

    #         if <COND_3>:
    t3 = min(t2, translate(<COND_3>))

    #             <OUTPUT_1> = <VALUE_1>
    <OUTPUT_1> = min(t3, <VALUE_1>)
    
    #             <OUTPUT_2> = <VALUE_2>
    <OUTPUT_2> = min(t3, <VALUE_2>)
    
    # else:
    t1 = min(t0, 1.0 - t1)
    
    #     if <COND_4>:
    t2 = min(t1, translate(<COND_4>))
    
    #         <OUTPUT_3> = <VALUE_3> 
    <OUTPUT_3> = min(t2, <VALUE_3>)

    #     else:
    t2 = min(t1, 1.0 - t2)

    #         if <COND_5>:
    t3 = min(t2, translate(<COND_5>))
    
    #             <OUTPUT_4> = <VALUE_4> 
    <OUTPUT_4> = min(t3, <VALUE_4>)

    #         else:
    t3 = min(t2, 1.0 - t3)

    #             <OUTPUT_1> = <VALUE_1> 
    <OUTPUT_1> = max(<OUTPUT_1>, min(t3, <VALUE_1>))

