## How to contribute?

This guide gives you information that aims to enable contributions to SPACE.
In Case you want to add a novel gpu compaction algorithm and compare it both to SPACE and Nvidia CUB several steps need to be done.

**Adding another compaction algorithm to the experiment framework**

In /src/benchmarks.cuh a launch function for each compaction algorithm is provided.
These are template functions (to allow flexible data types) following a naming scheme.
The return value of all benchmark functions is a self defined timings struct that contains measurements for each processing step.
There is also a struct intermediate_data that contains a number of pointers to intermediate data that is used either by CUB or SPACE. 
Measurements should be done with CUDA events.
Every function starts with bench and a unique ascending number followed by a short description of the algorithm.

Example:
bench1_base_variant is the first SPACE algo which is using a basic memory access pattern without any optimizations.

To add another algorithm the following steps need to be performed.

Add a new benchmark function with the same signature as the already present ones (to simplify adding new algorithms).
d_input is the input data array in gpu memory.
d_mask is the bitmask used for selection in gpu memory.
d_output is a buffer which stores the compaction results.

By using d_input and d_mask your algorithm needs to create a correct d_output array.
It is also needed to add measured run times of your kernels to the timings struct which is handed over as function parameter.

A CPU side validation of the compaction is done within the experiment framework and errors will be printed on the console.

After adding your benchmark in benchmark.cuh, it must also be added to main.cu.
In lines 228-252 all benchmarks that will be done are added to a data structure called benchs.
benchs is a vector of pairs. Each pair consists of a describing string and a timings struct (which is returned by calling your new experiment function).

Add your own benchmark function below line 252 following the same scheme.
If you run the benchmark suite, now your own algorithms will be included.

If you wish to add your algorithm to SPACE, create a merge request. Make sure that the CPU side validation of results completes without errors.

**Report issues or problems**

If you encounter bugs or problems, feel free to create a github issue.
Suggestions for addtional features can be creates as issue with the tag enhancement.

**Support**

In case you need support setting up or extending SPACE write me an email: Johannes.fett@tu-dresden.de
