# space_gpu

`SPACE` is a GPU centric C++ software for compaction experiments. It consists of data generators and a flexible experiment framework. Addtionally, scripts to visualize experiments are provided. For detailed information about compiling and running `SPACE`, see overview.pdf.
[overview.pdf](https://github.com/yogi-tud/space_gpu/blob/main/overview.pdf)

The binary takes a number of different runtime parameters that control the experiment.
See run_full.py for a comprehensive example running multiple experiments.

An overview of all parameters:
dataset , selectivity, datasize, cluster count, datatype

dataset controls the distribution of bits in the bit mask that is used to select items for write out.
dataset values: (0 = uniform, 1= single cluster, 2= multiple cluster)

selectivity as % of 1 bits in mask. Ranging from 0 to 1.

datasize in MiB of input data column. Mask will be generated accordingly.

cluster: number of clusters to distribute across the mask if dataset multiple cluster is picked

datatypes:  1-uint8 2-uint16 3-uint32 4-int 5-float 6-double

Device string: writes a device string back into csv experiment output and generates file names accordingly

subset of algorithms: (1:cub + space 8, 0: all)

Examples:

./gpu_compressstore2 0 0.25 1024 0 3 A100 0

This example runs with a uniform mask distribution and 25% selected elements. The input column is made of 1024 MiB uint32 elements. Output files are named with A100 as device and all algorithms will be peformed.

In CONTRIBUTE a guide how to contribute to the software can be found.
[contribute](https://github.com/yogi-tud/space_gpu/blob/main/CONTRIBUTE.md)
