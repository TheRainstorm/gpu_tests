#include <cstdio>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half
#include <type_traits> // For std::is_same_v

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if constexpr (std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if constexpr (std::is_same<T, double>::value) {
        AType = BType = CType = ComputeType = CUDA_R_64F;
    } else if constexpr (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else {
        printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));
    
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}

template<typename T>
void init_array(T* A, int M, int K) {
    for (int i = 0; i < M*K; ++i){
        if constexpr (std::is_same<T, int8_t>::value) {
            A[i] = static_cast<int8_t>(rand() % 256);
        } else if constexpr (std::is_same<T, __half>::value) {
            A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        } else {
            A[i] = static_cast<T>(rand()) / RAND_MAX;
        }
    }
}

template<typename T, typename S>
float bench_gemm(cublasHandle_t handle, int m, int n, int k, int warmup_times, int test_times, int algo=CUBLAS_GEMM_DEFAULT, bool rand_init=true) {
    T *A, *B; S *C;
    cudaMallocManaged(&A, m * k * sizeof(T));
    cudaMallocManaged(&B, k * n * sizeof(T));
    cudaMallocManaged(&C, m * n * sizeof(S));

    // 初始化数据
    init_array<T>(A, m, k);
    init_array<T>(B, k, n);
    S alpha = 1, beta = 0;
    
    auto perform_gemm = [&](){
        auto success = cublas_gemm_ex(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            B, A, C,  // 采用列主序模拟行主序
            n, k, n,
            &alpha, &beta, algo);
    };

    // 预热运行
    for (int i = 0; i < warmup_times; ++i) {
        perform_gemm();
    }

    // 创建计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;

    // 正式测试
    for (int t = 0; t < test_times; ++t) {
        cudaEventRecord(start);

        perform_gemm();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }

    // 计算性能
    float avg_time = total_time / test_times;
    float tflops = 2.0 * m * n * k / (avg_time * 1e-3) / 1e12;
    printf("M=%5d, N=%5d, K=%5d | Time: %6.3fms | TFLOPS: %6.2f\n",
           m, n, k, avg_time, tflops);

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avg_time;
}

int main(int argc, char *argv[]) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    for(int i = 0; i < deviceCount; ++i) {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: %s (SM %d.%d)\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    int device = 0; // 使用第一个设备
    bool use_TC = false; // 默认不使用 Tensor Core
    // 解析命令行参数
    switch (argc) {
        case 3:
            use_TC = atoi(argv[2]);
        case 2:
            device = atoi(argv[1]);
        case 1:
            break;
        default:
            printf("Usage: %s <device_id> <use_TC>\n", argv[0]);
            return 1;
    }

    printf("Select device: %d, use tensor core: %d\n", device, use_TC);
    cudaSetDevice(device);
    cudaGetDeviceProperties(&deviceProp, device);

    // 初始化cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 测试参数列表 (M, N, K)
    const std::vector<std::tuple<int, int, int>> test_cases = {
        {128, 1024, 4096},
        {128, 2048, 4096},
        {256, 1024, 4096},
        {256, 2048, 4096}
    };

    const int warmup_times = 10;
    const int test_times = 100;
    
    // 可以选择性地设置 Tensor Core 数学模式以利用 FP16 Tensor Cores (需要 SM 7.0+)
    if (deviceProp.major >= 7) {
        if (use_TC) {
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
            printf("Note: Tensor Core math mode enabled.\n");
        } else {
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
            printf("Note: Tensor Core is supported, but not enabled for test.\n");
        }
    } else {
         cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
         printf("Note: Device supports FP16 but not Tensor Cores for GEMM. Using default math mode.\n");
    }

    printf("\nBenchmarking GEMM...\n");

    // 测试 float
    printf("\n--- Testing float (FP32) ---\n");

    for (const auto& [M, N, K] : test_cases) {
        bench_gemm<float, float>(handle, M, N, K, warmup_times, test_times);
    }

    // 测试 double
    // 检查设备是否支持 double
    if (deviceProp.major >= 1) { // Double precision generally available on compute capability 1.x and higher
        printf("\n--- Testing double (FP64) ---\n");
         for (const auto& [M, N, K] : test_cases) {
            bench_gemm<double, double>(handle, M, N, K, warmup_times, test_times);
         }
    } else {
        printf("\n--- Skipping double (FP64) benchmark: Device does not support double precision ---\n");
    }

    // 测试 __half (FP16)
    // 检查设备是否支持 FP16 计算（通常需要 SM 5.3+ for storage, 6.0+ for operations, 7.0+ for Tensor Cores）
    if (deviceProp.major > 5 || (deviceProp.major == 5 && deviceProp.minor >= 3)) {
        printf("\n--- Testing __half (FP16) ---\n");
        for (const auto& [M, N, K] : test_cases) {
            bench_gemm<__half, __half>(handle, M, N, K, warmup_times, test_times);
        }
    } else {
         printf("\n--- Skipping __half (FP16) benchmark: Device does not support FP16 compute ---\n");
    }

    // 测试 int8_t
    if (deviceProp.major >= 6) { // SM 6.0+ for INT8
        printf("\n--- Testing int8_t (INT8) ---\n");
        for (const auto& [M, N, K] : test_cases) {
            bench_gemm<int8_t, int32_t>(handle, M, N, K, warmup_times, test_times);
        }
    } else {
         printf("\n--- Skipping int8_t (INT8) benchmark: Device does not support INT8 compute ---\n");
    }

    cublasDestroy(handle);

    return 0;
}