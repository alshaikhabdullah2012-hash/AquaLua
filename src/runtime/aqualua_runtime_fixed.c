/*
 * Aqualua Runtime - High-performance C backend for tensor operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

// Forward declarations
typedef struct AqualuaTensor AqualuaTensor;
typedef struct LinearLayer LinearLayer;

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

// Tensor structure
struct AqualuaTensor {
    void* data;
    int* shape;
    int ndim;
    DataType dtype;
    DeviceType device;
    size_t size;
    size_t bytes;
    bool requires_grad;
    AqualuaTensor* grad;
    int ref_count;
};

// Linear layer structure
struct LinearLayer {
    AqualuaTensor* weight;
    AqualuaTensor* bias;
    int in_features;
    int out_features;
};

// Memory management
void* aqualua_malloc(size_t size, DeviceType device) {
    return malloc(size);
}

void aqualua_free(void* ptr, DeviceType device) {
    free(ptr);
}

// Tensor creation
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
    
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
    
    size_t element_size = sizeof(float);
    tensor->bytes = tensor->size * element_size;
    tensor->data = aqualua_malloc(tensor->bytes, device);
    
    return tensor;
}

void aqualua_tensor_destroy(AqualuaTensor* tensor) {
    if (!tensor) return;
    aqualua_free(tensor->data, tensor->device);
    free(tensor->shape);
    free(tensor);
}

AqualuaTensor* aqualua_tensor_zeros(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    memset(tensor->data, 0, tensor->bytes);
    return tensor;
}

AqualuaTensor* aqualua_tensor_ones(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) {
        data[i] = 1.0f;
    }
    return tensor;
}

AqualuaTensor* aqualua_tensor_random(int* shape, int ndim, DataType dtype, DeviceType device) {
    AqualuaTensor* tensor = aqualua_tensor_create(shape, ndim, dtype, device);
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    return tensor;
}

// Tensor operations
AqualuaTensor* aqualua_tensor_add(AqualuaTensor* a, AqualuaTensor* b) {
    AqualuaTensor* result = aqualua_tensor_create(a->shape, a->ndim, a->dtype, a->device);
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* result_data = (float*)result->data;
    
    for (size_t i = 0; i < a->size; i++) {
        result_data[i] = a_data[i] + b_data[i];
    }
    return result;
}

AqualuaTensor* aqualua_tensor_multiply(AqualuaTensor* a, AqualuaTensor* b) {
    AqualuaTensor* result = aqualua_tensor_create(a->shape, a->ndim, a->dtype, a->device);
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* result_data = (float*)result->data;
    
    for (size_t i = 0; i < a->size; i++) {
        result_data[i] = a_data[i] * b_data[i];
    }
    return result;
}

AqualuaTensor* aqualua_tensor_matmul(AqualuaTensor* a, AqualuaTensor* b) {
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    
    int result_shape[2] = {m, n};
    AqualuaTensor* result = aqualua_tensor_create(result_shape, 2, a->dtype, a->device);
    
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* c_data = (float*)result->data;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    return result;
}

AqualuaTensor* aqualua_tensor_relu(AqualuaTensor* input) {
    AqualuaTensor* result = aqualua_tensor_create(input->shape, input->ndim, input->dtype, input->device);
    float* input_data = (float*)input->data;
    float* result_data = (float*)result->data;
    
    for (size_t i = 0; i < input->size; i++) {
        result_data[i] = fmaxf(0.0f, input_data[i]);
    }
    return result;
}

AqualuaTensor* aqualua_tensor_sigmoid(AqualuaTensor* input) {
    AqualuaTensor* result = aqualua_tensor_create(input->shape, input->ndim, input->dtype, input->device);
    float* input_data = (float*)input->data;
    float* result_data = (float*)result->data;
    
    for (size_t i = 0; i < input->size; i++) {
        result_data[i] = 1.0f / (1.0f + expf(-input_data[i]));
    }
    return result;
}

// Linear layer
LinearLayer* aqualua_linear_create(int in_features, int out_features, DeviceType device) {
    LinearLayer* layer = malloc(sizeof(LinearLayer));
    layer->in_features = in_features;
    layer->out_features = out_features;
    
    int weight_shape[2] = {out_features, in_features};
    layer->weight = aqualua_tensor_random(weight_shape, 2, DTYPE_F32, device);
    
    int bias_shape[1] = {out_features};
    layer->bias = aqualua_tensor_zeros(bias_shape, 1, DTYPE_F32, device);
    
    return layer;
}

AqualuaTensor* aqualua_linear_forward(LinearLayer* layer, AqualuaTensor* input) {
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
    printf("Aqualua Runtime initialized\n");
}

void aqualua_device_cleanup() {
    printf("Aqualua Runtime cleaned up\n");
}

DeviceType aqualua_device_auto_select() {
    return DEVICE_CPU;
}

// Python interface
__declspec(dllexport) AqualuaTensor* py_tensor_create(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_create(shape, ndim, (DataType)dtype, (DeviceType)device);
}

__declspec(dllexport) AqualuaTensor* py_tensor_zeros(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_zeros(shape, ndim, (DataType)dtype, (DeviceType)device);
}

__declspec(dllexport) AqualuaTensor* py_tensor_ones(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_ones(shape, ndim, (DataType)dtype, (DeviceType)device);
}

__declspec(dllexport) AqualuaTensor* py_tensor_random(int* shape, int ndim, int dtype, int device) {
    return aqualua_tensor_random(shape, ndim, (DataType)dtype, (DeviceType)device);
}

__declspec(dllexport) AqualuaTensor* py_tensor_add(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_add(a, b);
}

__declspec(dllexport) AqualuaTensor* py_tensor_multiply(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_multiply(a, b);
}

__declspec(dllexport) AqualuaTensor* py_tensor_matmul(AqualuaTensor* a, AqualuaTensor* b) {
    return aqualua_tensor_matmul(a, b);
}

__declspec(dllexport) AqualuaTensor* py_tensor_relu(AqualuaTensor* input) {
    return aqualua_tensor_relu(input);
}

__declspec(dllexport) AqualuaTensor* py_tensor_sigmoid(AqualuaTensor* input) {
    return aqualua_tensor_sigmoid(input);
}

__declspec(dllexport) LinearLayer* py_linear_create(int in_features, int out_features, int device) {
    return aqualua_linear_create(in_features, out_features, (DeviceType)device);
}

__declspec(dllexport) AqualuaTensor* py_linear_forward(LinearLayer* layer, AqualuaTensor* input) {
    return aqualua_linear_forward(layer, input);
}

__declspec(dllexport) void py_tensor_destroy(AqualuaTensor* tensor) {
    aqualua_tensor_destroy(tensor);
}

__declspec(dllexport) void py_linear_destroy(LinearLayer* layer) {
    aqualua_linear_destroy(layer);
}

__declspec(dllexport) void py_device_init() {
    aqualua_device_init();
}

__declspec(dllexport) void py_device_cleanup() {
    aqualua_device_cleanup();
}

__declspec(dllexport) int py_device_auto_select() {
    return (int)aqualua_device_auto_select();
}

__declspec(dllexport) float* py_tensor_data_f32(AqualuaTensor* tensor) {
    return (float*)tensor->data;
}

__declspec(dllexport) int* py_tensor_shape(AqualuaTensor* tensor) {
    return tensor->shape;
}

__declspec(dllexport) int py_tensor_ndim(AqualuaTensor* tensor) {
    return tensor->ndim;
}

__declspec(dllexport) size_t py_tensor_size(AqualuaTensor* tensor) {
    return tensor->size;
}