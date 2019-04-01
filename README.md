# Scanner: Efficient Video Analysis at Scale [![GitHub tag](https://img.shields.io/github/tag/scanner-research/scanner.svg)](https://GitHub.com/scanner-research/scanner/tags/) [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for developing applications that efficiently process large video datasets. 

To learn more about Scanner, see the documentation at [scanner.run](http://scanner.run), check out the [various example applications](https://github.com/scanner-research/scanner/tree/master/examples), or read the SIGGRAPH 2018 Technical Paper: "[Scanner: Efficient Video Analysis at Scale](http://graphics.stanford.edu/papers/scanner/)".

## Documentation

Scanner's documentation is hosted at [scanner.run](http://scanner.run). Here
are a few links to get you started:

* [Installation](http://scanner.run/installation.html)
* [Getting Started](http://scanner.run/getting-started.html)
* [Programming Handbook](http://scanner.run/programming-handbook.html)
* [API Reference](http://scanner.run/api.html)
* [SIGGRAPH 2018 Technical Paper](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)
* [Scanner Examples](https://github.com/scanner-research/scanner/tree/master/examples)

## Contributing

If you'd like to contribute to the development of Scanner, you should first
build Scanner [from source](http://scanner.run/from_source.html).

Please submit a pull-request rebased against the most recent version of the
master branch and we will review your changes to be merged. Thanks for
contributing!

### Running tests
You can run the full suite of tests by executing `make test` in the directory
you used to build Scanner. This will run both the C++ tests and the end-to-end
tests that verify the python API.

## About
Scanner is an active research project, part of a collaboration between Stanford and Carnegie Mellon University. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

Scanner was developed with the support of the NSF (IIS-1539069), the Intel Corporation (through the Intel Science and Technology Center for Visual Cloud Computing and the NSF/Intel VEC program), and by Google.

### Paper citation
Scanner was published at SIGGRAPH 2018 as "[Scanner: Efficient Video Analysis at Scale](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)" by Poms, Crichton, Hanrahan, and Fatahalian. If you use Scanner in your research, we'd appreciate it if you cite the paper with the following bibtex:
```
@article{Poms:2018:Scanner,
 author = {Poms, Alex and Crichton, Will and Hanrahan, Pat and Fatahalian, Kayvon},
 title = {Scanner: Efficient Video Analysis at Scale},
 journal = {ACM Trans. Graph.},
 issue_date = {August 2018},
 volume = {37},
 number = {4},
 month = jul,
 year = {2018},
 issn = {0730-0301},
 pages = {138:1--138:13},
 articleno = {138},
 numpages = {13},
 url = {http://doi.acm.org/10.1145/3197517.3201394},
 doi = {10.1145/3197517.3201394},
 acmid = {3201394},
 publisher = {ACM},
 address = {New York, NY, USA},
} 
```

