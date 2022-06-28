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
  - name: Christian Schwarz 
    affiliation: TU Dresden
  - name: Dirk Habich
    affiliation: TU Dresden
  - name: Wolfgang Lehner
    affiliation: TU Dresden
date: 28 June 2022
bibliography: bib.bib

# Summary

Compacting is a common and heavily used operation in different application areas like statistics, database systems, simulations and
artificial intelligence. The task of this operation is to produce a smaller output array by writing selected elements of an input array contiguously
back to a new output array. The selected elements are usually defined by means of a bit mask. With the always increasing amount of data
elements to be processed in the different application areas, better performance becomes a key factor for this operation. Thus, exploiting the
parallel capabilities of GPUs to speed up the compacting operation is of great interest. We introduce smart partitioning for GPU compaction (`SPACE` ) as a set of different optimization approaches for GPUs. An detailed guide of setting up and using the software is found in the file overview.pdf. paper.pdf contains the published Euro-par 2022 paper, in which `SPACE`  is described in great detail.

# Statement of need

`SPACE` is a GPU centric C++ software for compaction experiments. It consists of different data generators and a flexible experiment framework.
8 different `SPACE` Variants can be compared against the NVIDIA supplied CUB library for GPU compaction. Data type, percentage of selected data and data distrubtions are modifiable as execution parameter for the generated C++ binary. Different Python runscripts for performing sets of experiments and reproducing the experiments from our paper are provided. Output of experiments is written as csv files. For visualizing the results Python scripts based on Matplotlib are also provided. 

`SPACE` was designed to allow researchers to evaluate compaction algorithms against a solid baseline across a variety of input data. It can be modified by adding additional compaction algorithms. It outperforms the current state-of-the-art `[@cub]`. Notable research about compaction has been performed by `[@bakunas2017efficient]`, who classify compaction on GPU into the two categories "prefix sum based" and "atomic based". `SPACE` is a pre fix sum based approach.

# Acknowledgements

This work is funded by the German Research Foundation within the RTG 1907 (RoSI) as well as by the European Union's Horizon 2020 research and innovative program under grant agreement number 957407 (DAPHNE project).

# References
`[@cub]`
`[@bakunas2017efficient]`
