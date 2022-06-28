#ifndef CUB_WRAPS_CUH
#define CUB_WRAPS_CUH

#include <cstdint>
#include <iterator>
#include <cub/cub.cuh>

#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "kernels/kernel_3pass.cuh"

struct bitstream_iterator {
    uint8_t* bytepointer;
    size_t idx;

    __host__ __device__ bitstream_iterator(uint8_t* bytepointer, size_t idx = 0) : bytepointer(bytepointer), idx(idx)
    {
    }

    __host__ __device__ bool operator[](size_t i)
    {
        uint8_t byte = bytepointer[(idx + i) >> 3];
        return byte >> (7 - (idx + i) % 8) & 0b1;
    }

    __host__ __device__ bitstream_iterator operator+(size_t i)
    {
        return bitstream_iterator{bytepointer, idx + i};
    }

    __host__ __device__ bitstream_iterator operator-(size_t i)
    {
        return bitstream_iterator{bytepointer, idx - i};
    }

    __host__ __device__ bool operator++()
    {
        idx++;
        return this->operator[](0);
    }

    __host__ __device__ bool operator--()
    {
        idx--;
        return this->operator[](0);
    }
};
template <> struct std::iterator_traits<bitstream_iterator> {
    typedef bool value_type;
};

float launch_cub_pss(cudaStream_t stream, cudaEvent_t ce_start, cudaEvent_t ce_stop, uint32_t* d_pss, uint32_t* d_pss_total, uint32_t chunk_count)
{
    if (!chunk_count) return 0;
    // use cub pss for now
    launch_3pass_pssskip(stream, d_pss, d_pss_total, chunk_count);
    uint32_t* d_pss_tmp;
    CUDA_TRY(cudaMalloc(&d_pss_tmp, chunk_count * sizeof(uint32_t)));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count));
    CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // timed relevant computation
    float time = 0;
    if (stream == 0) {
        CUDA_TIME(
            ce_start, ce_stop, 0, &time,
            CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count, 0)));
    }
    else {
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count, stream));
        time = 0;
    }
    CUDA_TRY(cudaFree(d_temp_storage));
    CUDA_TRY(cudaMemcpyAsync(d_pss, d_pss_tmp, chunk_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_TRY(cudaFree(d_pss_tmp));
    launch_3pass_pssskip(stream, d_pss, d_pss_total, chunk_count);
    return time;
}

template <typename T>
float launch_cub_flagged_biterator(
    cudaEvent_t ce_start, cudaEvent_t ce_stop, T* d_input, T* d_output, uint8_t* d_mask, uint32_t* d_selected_out, uint32_t element_count)
{
    bitstream_iterator bit{d_mask};
    // determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input, bit, d_output, d_selected_out, element_count);
    // allocate temporary storage
    CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run selection
    float time = 0;
    CUDA_TIME(
        ce_start, ce_stop, 0, &time,
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input, bit, d_output, d_selected_out, element_count));
    CUDA_TRY(cudaFree(d_temp_storage));
    return time;
}

template <typename T>
float launch_cub_flagged_bytemask(
    cudaEvent_t ce_start, cudaEvent_t ce_stop, T* d_input, T* d_output, uint8_t* h_mask, uint32_t* d_selected_out, uint32_t element_count)
{
    uint8_t* h_bytemask = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * element_count));
    uint8_t* d_bytemask;
    CUDA_TRY(cudaMalloc(&d_bytemask, sizeof(uint8_t) * element_count));
    // sanitize bitmask to bytemask
    for (int i = 0; i < element_count / 8; i++) {
        uint32_t acc = reinterpret_cast<uint8_t*>(h_mask)[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i * 8 + (7 - j);
            bool v = 0b1 & (acc >> j);
            h_bytemask[idx] = v;
        }
    }
    // copy bytemask to gpu
    CUDA_TRY(cudaMemcpy(d_bytemask, h_bytemask, sizeof(uint8_t) * element_count, cudaMemcpyHostToDevice));
    // determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input, d_bytemask, d_output, d_selected_out, element_count);
    // allocate temporary storage
    CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run selection
    float time = 0;
    CUDA_TIME(
        ce_start, ce_stop, 0, &time,
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_input, d_bytemask, d_output, d_selected_out, element_count));
    // free bytemask from gpu
    CUDA_TRY(cudaFree(d_bytemask));
    CUDA_TRY(cudaFree(d_temp_storage));
    free(h_bytemask);

    return time;
}

#endif
