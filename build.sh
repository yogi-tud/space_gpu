cmake .
cmake --build . --target clean -- -j 12
cmake --build . --target gpu_compressstore2 -- -j 12

