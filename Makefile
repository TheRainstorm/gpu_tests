
GENCODE = -gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_61,code=sm_61

# CUDA_CXX_FLAGS = -std=c++17 -O3 -Xcompiler -fopenmp ${GENCODE} -lcublas_static -lcublasLt_static -lculibos
CUDA_CXX_FLAGS = -std=c++17 -O3 -Xcompiler -fopenmp ${GENCODE} -lcublas

all: gemm_bench gemm_test
gemm_bench: gemm_bench.cu
	nvcc $< ${CUDA_CXX_FLAGS} -o $@

gemm_test: gemm_test.cu
	nvcc $< ${CUDA_CXX_FLAGS} -o $@

.PHONY: all