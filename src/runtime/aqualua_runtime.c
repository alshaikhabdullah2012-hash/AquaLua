/*
 * Aqualua Runtime - High-performance C backend for tensor operations
 * This provides the core runtime system for Aqualua with direct hardware access
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

// Forward declarations
typedef struct AqualuaTensor AqualuaTensor;
typedef struct AqualuaModel AqualuaModel;
typedef struct AqualuaDevice AqualuaDevice;

// Device types
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_GPU = 1,
    DEVICE_AUTO = 2
} DeviceType;

// Data types
typedef enum {
    DTYPE_F32 = 0,
    DTYPE_F64 = 1,
    DTYPE_F16 = 2,
    DTYPE_I32 = 3,
    DTYPE_I64 = 4,
    DTYPE_BOOL = 5
} DataType;

// Tensor structure - core data type
struct AqualuaTensor {
    void* data;              // Raw data pointer
    int* shape;              // Shape array
    int ndim;                // Number of dimensions
    DataType dtype;          // Data type
    DeviceType device;       // Device location
    size_t size;             // Total number of elements
    size_t bytes;            // Total bytes
    bool requires_grad;      // For autodiff
    AqualuaTensor* grad;     // Gradient tensor
    int ref_count;           // Reference counting for memory management
};

// Device management
struct AqualuaDevice {
    DeviceType type;
    int device_id;
    void* context;           // CUDA context or CPU thread pool
    size_t memory_used;
    size_t memory_limit;
};

// Model structure
struct AqualuaModel {
    char* name;
    AqualuaTensor** parameters;
    int num_parameters;
    void (*forward)(AqualuaModel*, AqualuaTensor*, AqualuaTensor*);
    DeviceType device;
};

// Global device manager
static AqualuaDevice* current_device = NULL;
static AqualuaDevice cpu_device = {DEVICE_CPU, 0, NULL, 0, SIZE_MAX};

#ifdef __CUDA_ARCH__
static AqualuaDevice gpu_device = {DEVICE_GPU, 0, NULL, 0, 0};
static cublasHandle_t cublas_handle;
static cudnnHandle_t cudnn_handle;
#endif

// Memory management functions
void* aqualua_malloc(size_t size, DeviceType device) {
    void* ptr = NULL;
    
    switch (device) {
        case DEVICE_CPU:
            ptr = malloc(size);
            break;
            
        case DEVICE_GPU:
#ifdef __CUDA_ARCH__
            cudaMalloc(&ptr, size);
#else
            fprintf(stderr, "GPU not available, falling back to CPU\n");
            ptr = malloc(size);
#endif
            break;
            
        default:
            ptr = malloc(size);
    }
    
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed for %zu bytes\n", size);
        exit(1);
    }
    
    return ptr;
}

void aqualua_free(void* ptr, DeviceType device) {
    if (!ptr) return;
    
    switch (device) {
        case DEVICE_CPU:
            free(ptr);
            break;
            
        case DEVICE_GPU:
#ifdef __CUDA_ARCH__
            cudaFree(ptr);
#else
            free(ptr);
#endif
            break;
            
        default:
            free(ptr);
    }
}

// Tensor creation functions
AqualuaTensor* aqualua_tensor_create(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = malloc(sizeof(AqualuaTensor));
    
    tensor->shape = malloc(ndim * sizeof(int));
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->requires_grad = false;
    tensor->grad = NULL;
    tensor->ref_count = 1;
    
    // Calculate size
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
    
    // Calculate bytes based on dtype
    size_t element_size;
    switch (dtype) {
        case DTYPE_F32: element_size = sizeof(float); break;
        case DTYPE_F64: element_size = sizeof(double); break;
        case DTYPE_F16: element_size = 2; break;
        case DTYPE_I32: element_size = sizeof(int); break;
        case DTYPE_I64: element_size = sizeof(long long); break;
        case DTYPE_BOOL: element_size = sizeof(bool); break;
        default: element_size = sizeof(float);
    }
    
    tensor->bytes = tensor->size * element_size;
    tensor->data = aqualua_malloc(tensor->bytes, device);
    
    return tensor;
}

void aqualua_tensor_destroy(AqualuaTensor* tensor) {
    if (!tensor) return;
    
    tensor->ref_count--;
    if (tensor->ref_count > 0) return;
    
    aqualua_free(tensor->data, tensor->device);
    free(tensor->shape);
    
    if (tensor->grad) {
        aqualua_tensor_destroy(tensor->grad);
    }
    
    free(tensor);
}

AqualuaTensor* aqualua_tensor_zeros(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    
    if (device == DEVICE_CPU) {
        memset(tensor->data, 0, tensor->bytes);
    } else {
#ifdef __CUDA_ARCH__
        cudaMemset(tensor->data, 0, tensor->bytes);
#else
        memset(tensor->data, 0, tensor->bytes);
#endif
    }
    
    return tensor;
}

AqualuaTensor* aqualua_tensor_ones(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    
    if (device == DEVICE_CPU && dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = 1.0f;
        }
    }
    // TODO: GPU implementation
    
    return tensor;
}

AqualuaTensor* aqualua_tensor_random(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    
    if (device == DEVICE_CPU && dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random [-1, 1]
        }
    }
    
    return tensor;
}

// Core tensor operations
AqualuaTensor* aqualua_tensor_add(AqualuaTensor* a, AqualuaTensor* b) {
    assert(a->size == b->size && "Tensor size mismatch");
    assert(a->dtype == b->dtype && "Tensor dtype mismatch");
    
    AqualuaTensor* result = aqualua_tensor_create(a->shape, a->ndim, a->dtype, a->device);
    
    if (a->device == DEVICE_CPU && a->dtype == DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* result_data = (float*)result->data;
        
        // Vectorized addition using compiler auto-vectorization
        for (size_t i = 0; i < a->size; i++) {
            result_data[i] = a_data[i] + b_data[i];
        }
    }
    
    return result;
}

AqualuaTensor* aqualua_tensor_multiply(AqualuaTensor* a, AqualuaTensor* b) {
    assert(a->size == b->size && "Tensor size mismatch");
    assert(a->dtype == b->dtype && "Tensor dtype mismatch");
    
    AqualuaTensor* result = aqualua_tensor_create(a->shape, a->ndim, a->dtype, a->device);
    
    if (a->device == DEVICE_CPU && a->dtype == DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->size; i++) {
            result_data[i] = a_data[i] * b_data[i];
        }
    }
    
    return result;
}

// Matrix multiplication - the heart of ML
AqualuaTensor* aqualua_tensor_matmul(AqualuaTensor* a, AqualuaTensor* b) {
    assert(a->ndim >= 2 && b->ndim >= 2 && "Matrices must be at least 2D");
    assert(a->shape[a->ndim-1] == b->shape[b->ndim-2] && "Matrix dimension mismatch");
    
    int m = a->shape[a->ndim-2];
    int k = a->shape[a->ndim-1];
    int n = b->shape[b->ndim-1];
    
    int result_shape[2] = {m, n};
    AqualuaTensor* result = aqualua_tensor_create(result_shape, 2, a->dtype, a->device);
    
    if (a->device == DEVICE_CPU && a->dtype == DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* c_data = (float*)result->data;
        
        // Optimized matrix multiplication with blocking for cache efficiency
        const int BLOCK_SIZE = 64;
        
        for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
                    
                    int i_max = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;
                    int j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
                    int k_max = (kk + BLOCK_SIZE < k) ? kk + BLOCK_SIZE : k;
                    
                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j++) {
                            float sum = 0.0f;
                            for (int l = kk; l < k_max; l++) {
                                sum += a_data[i * k + l] * b_data[l * n + j];
                            }
                            c_data[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }
#ifdef __CUDA_ARCH__
    else if (a->device == DEVICE_GPU && a->dtype == DTYPE_F32) {
        // Use cuBLAS for GPU matrix multiplication
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   n, m, k,
                   &alpha,
                   (float*)b->data, n,
                   (float*)a->data, k,
                   &beta,
                   (float*)result->data, n);
    }
#endif
    
    return result;
}

// Activation functions
AqualuaTensor* aqualua_tensor_relu(AqualuaTensor* input) {
    AqualuaTensor* result = aqualua_tensor_create(input->shape, input->ndim, input->dtype, input->device);
    
    if (input->device == DEVICE_CPU && input->dtype == DTYPE_F32) {
        float* input_data = (float*)input->data;
        float* result_data = (float*)result->data;
        
        for (size_t i = 0; i < input->size; i++) {
            result_data[i] = fmaxf(0.0f, input_data[i]);
        }
    }
    
    return result;
}

AqualuaTensor* aqualua_tensor_sigmoid(AqualuaTensor* input) {
    AqualuaTensor* result = aqualua_tensor_create(input->shape, input->ndim, input->dtype, input->device);
    
    if (input->device == DEVICE_CPU && input->dtype == DTYPE_F32) {
        float* input_data = (float*)input->data;
        float* result_data = (float*)result->data;
        
        for (size_t i = 0; i < input->size; i++) {
            result_data[i] = 1.0f / (1.0f + expf(-input_data[i]));
        }
    }
    
    return result;
}

// Loss functions
AqualuaTensor* aqualua_cross_entropy_loss(AqualuaTensor* predictions, AqualuaTensor* targets) {
    assert(predictions->size == targets->size && "Prediction and target size mismatch");
    
    int result_shape[1] = {1};
    AqualuaTensor* result = aqualua_tensor_create(result_shape, 1, DTYPE_F32, predictions->device);
    
    if (predictions->device == DEVICE_CPU && predictions->dtype == DTYPE_F32) {
        float* pred_data = (float*)predictions->data;
        float* target_data = (float*)targets->data;
        float* result_data = (float*)result->data;
        
        float loss = 0.0f;
        for (size_t i = 0; i < predictions->size; i++) {
            loss += -target_data[i] * logf(pred_data[i] + 1e-8f); // Add epsilon for numerical stability
        }
        
        result_data[0] = loss / predictions->shape[0]; // Average over batch
    }
    
    return result;
}

// Neural network layers
typedef struct {
    AqualuaTensor* weight;
    AqualuaTensor* bias;
    int in_features;
    int out_features;
} LinearLayer;

LinearLayer* aqualua_linear_create(int in_features, int out_features, DeviceType device) {
    LinearLayer* layer = malloc(sizeof(LinearLayer));
    layer->in_features = in_features;
    layer->out_features = out_features;
    
    // Initialize weights with Xavier initialization
    int weight_shape[2] = {out_features, in_features};
    layer->weight = aqualua_tensor_random(weight_shape, 2, DTYPE_F32, device);
    
    // Scale weights for Xavier initialization
    if (device == DEVICE_CPU) {
        float* weight_data = (float*)layer->weight->data;
        float scale = sqrtf(2.0f / (in_features + out_features));
        for (size_t i = 0; i < layer->weight->size; i++) {
            weight_data[i] *= scale;
        }
    }
    
    // Initialize bias to zero
    int bias_shape[1] = {out_features};
    layer->bias = aqualua_tensor_zeros(bias_shape, 1, DTYPE_F32, device);
    
    return layer;
}

AqualuaTensor* aqualua_linear_forward(LinearLayer* layer, AqualuaTensor* input) {
    // Linear transformation: output = input @ weight.T + bias
    AqualuaTensor* output = aqualua_tensor_matmul(input, layer->weight);
    AqualuaTensor* result = aqualua_tensor_add(output, layer->bias);
    
    aqualua_tensor_destroy(output);
    return result;
}

void aqualua_linear_destroy(LinearLayer* layer) {
    if (!layer) return;
    
    aqualua_tensor_destroy(layer->weight);
    aqualua_tensor_destroy(layer->bias);
    free(layer);
}

// Device management
void aqualua_device_init() {
    current_device = &cpu_device;
    
#ifdef __CUDA_ARCH__
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 0) {
        cudaSetDevice(0);
        cublasCreate(&cublas_handle);
        cudnnCreate(&cudnn_handle);
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        gpu_device.memory_limit = total_mem;
        
        printf("Aqualua: GPU initialized with %zu MB memory\n", total_mem / (1024 * 1024));
    }
#endif
    
    printf("Aqualua Runtime initialized\n");
}

void aqualua_device_cleanup() {
#ifdef __CUDA_ARCH__
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
#endif
    printf("Aqualua Runtime cleaned up\n");
}

DeviceType aqualua_device_auto_select() {
#ifdef __CUDA_ARCH__
    int device_count;
    cudaGetDeviceCount(&device_count);
    return (device_count > 0) ? DEVICE_GPU : DEVICE_CPU;
#else
    return DEVICE_CPU;
#endif
}

// Training utilities
typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int step_count;
} AdamOptimizer;

AdamOptimizer* aqualua_adam_create(float lr, float beta1, float beta2) {
    AdamOptimizer* optimizer = malloc(sizeof(AdamOptimizer));
    optimizer->learning_rate = lr;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = 1e-8f;
    optimizer->step_count = 0;
    
    return optimizer;
}

void aqualua_adam_step(AdamOptimizer* optimizer, AqualuaTensor* param, AqualuaTensor* grad) {
    // Simplified Adam optimizer step
    // In production, this would maintain momentum buffers per parameter
    
    optimizer->step_count++;
    
    if (param->device == DEVICE_CPU && param->dtype == DTYPE_F32) {
        float* param_data = (float*)param->data;
        float* grad_data = (float*)grad->data;
        
        float lr_corrected = optimizer->learning_rate * 
                           sqrtf(1.0f - powf(optimizer->beta2, optimizer->step_count)) /
                           (1.0f - powf(optimizer->beta1, optimizer->step_count));
        
        for (size_t i = 0; i < param->size; i++) {
            param_data[i] -= lr_corrected * grad_data[i];
        }
    }
}

// Python FFI interface functions
#ifdef __cplusplus
extern "C" {
#endif
    // Export functions for Python to call
    __declspec(dllexport) AqualuaTensor* py_tensor_create(int* shape, int ndim, int dtype, int device);
    __declspec(dllexport) AqualuaTensor* py_tensor_zeros(int* shape, int ndim, int dtype, int device);
    __declspec(dllexport) AqualuaTensor* py_tensor_ones(int* shape, int ndim, int dtype, int device);
    __declspec(dllexport) AqualuaTensor* py_tensor_random(int* shape, int ndim, int dtype, int device);
    
    __declspec(dllexport) AqualuaTensor* py_tensor_add(AqualuaTensor* a, AqualuaTensor* b);
    __declspec(dllexport) AqualuaTensor* py_tensor_multiply(AqualuaTensor* a, AqualuaTensor* b);
    __declspec(dllexport) AqualuaTensor* py_tensor_matmul(AqualuaTensor* a, AqualuaTensor* b);
    __declspec(dllexport) AqualuaTensor* py_tensor_relu(AqualuaTensor* input);
    __declspec(dllexport) AqualuaTensor* py_tensor_sigmoid(AqualuaTensor* input);
    
    __declspec(dllexport) LinearLayer* py_linear_create(int in_features, int out_features, int device);
    __declspec(dllexport) AqualuaTensor* py_linear_forward(LinearLayer* layer, AqualuaTensor* input);
    
    __declspec(dllexport) void py_tensor_destroy(AqualuaTensor* tensor);
    __declspec(dllexport) void py_linear_destroy(LinearLayer* layer);
    
    __declspec(dllexport) void py_device_init();
    __declspec(dllexport) void py_device_cleanup();
    __declspec(dllexport) int py_device_auto_select();
    
    // Data access for Python
    __declspec(dllexport) float* py_tensor_data_f32(AqualuaTensor* tensor);
    __declspec(dllexport) int* py_tensor_shape(AqualuaTensor* tensor);
    __declspec(dllexport) int py_tensor_ndim(AqualuaTensor* tensor);
    __declspec(dllexport) size_t py_tensor_size(AqualuaTensor* tensor);
}

// Implementation of Python interface
AqualuaTensor* py_tensor_create(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_create(shape, ndim, (DataType)dtype, (DeviceType)device);
}

AqualuaTensor* py_tensor_zeros(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_zeros(shape, ndim, (DataType)dtype, (DeviceType)device);
}

AqualuaTensor* py_tensor_ones(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_ones(shape, ndim, (DataType)dtype, (DeviceType)device);
}

AqualuaTensor* py_tensor_random(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_random(shape, ndim, (DataType)dtype, (DeviceType)device);
}

AqualuaTensor* py_tensor_add(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_add(a, b);
}

AqualuaTensor* py_tensor_multiply(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_multiply(a, b);
}

AqualuaTensor* py_tensor_matmul(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_matmul(a, b);
}

AqualuaTensor* py_tensor_relu(AqualuaTensor* input) {
    return aqualua_tensor_relu(input);
}

AqualuaTensor* py_tensor_sigmoid(AqualuaTensor* input) {
    return aqualua_tensor_sigmoid(input);
}

LinearLayer* py_linear_create(int in_features, int out_features, int device) {
    return aqualua_linear_create(in_features, out_features, (DeviceType)device);
}

AqualuaTensor* py_linear_forward(LinearLayer* layer, AqualuaTensor* input) {
    return aqualua_linear_forward(layer, input);
}

void py_tensor_destroy(AqualuaTensor* tensor) {
    aqualua_tensor_destroy(tensor);
}

void py_linear_destroy(LinearLayer* layer) {
    aqualua_linear_destroy(layer);
}

void py_device_init() {
    aqualua_device_init();
}

void py_device_cleanup() {
    aqualua_device_cleanup();
}

int py_device_auto_select() {
    return (int)aqualua_device_auto_select();
}

float* py_tensor_data_f32(AqualuaTensor* tensor) {
    return (float*)tensor->data;
}

int* py_tensor_shape(AqualuaTensor* tensor) {
    return tensor->shape;
}

int py_tensor_ndim(AqualuaTensor* tensor) {
    return tensor->ndim;
}

size_t py_tensor_size(AqualuaTensor* tensor) {
    return tensor->size;
}

#ifdef __cplusplus
}
#endif