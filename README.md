---
title: 'Accelerating Parallel Operation for Compacting Selected Elements on GPUs'
tags:
  - Compacting
  - GPU
  - dynamics
  - Optimization
  - Parallel
authors:
  - name: Johannes Fett
    orcid: 0000-0001-7898-0502 
    affiliation: TU Dresden
    corresponding: true
  - name: Urs Kober 
    affiliation: TU Dresden
date: 13 August 2017
bibliography: bib.bib

# Summary

Compacting is a common and heavily used operation in different application areas like statistics, database systems, simulations and
artificial intelligence. The task of this operation is to produce a smaller output array by writing selected elements of an input array contiguously
back to a new output array. The selected elements are usually defined by means of a bit mask. With the always increasing amount of data
elements to be processed in the different application areas, better performance becomes a key factor for this operation. Thus, exploiting the
parallel capabilities of GPUs to speed up the compacting operation is of great interest. We introduce smart partitioning for GPU compaction (`SPACE` ) as a set of different optimization approaches for GPUs. An detailed guide of setting up and using the software is found in the file overview.pdf. paper.pdf contains the published Euro-par 2022 paper, in which `SPACE`  is described in great detail.

# Statement of need

`SPACE` is a GPU centric C++ software for compaction experiments. It consists of different data generators and a flexible experiment framework.
8 different `SPACE` Variants can be compared against the NVIDIA supplied CUB library for GPU compaction. Data type, percentage of selected data and data distrubtions are modifiable as execution parameter for the generated C++ binary. Different Python runscripts for performing sets of experiments are  provided. Output of experiments is written as csv files. For visualizing the results Python scripts based on Matplotlib are also provided. 

`SPACE` was designed to allow researchers to evaluate compaction algorithms against a solid baseline across a variety of input data. It can be modified by adding additional compaction algorithms. It outperforms the current state-of-the-art `[@cub]`.


is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Acknowledgements

This work is funded by the German Research Foundation within the RTG 1907 (RoSI) as well as by the European Union's Horizon 2020 research and innovative program under grant agreement number 957407 (DAPHNE project).

# References
