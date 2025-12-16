/*
 * AquaLua Python Bridge - Universal Python Library Integration
 * Embeds Python interpreter to access ALL Python libraries
 */

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Python bridge initialization
static int python_initialized = 0;

void aqualua_init_python() {
    if (!python_initialized) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            fprintf(stderr, "Failed to initialize Python\n");
            return;
        }
        
        // Add current directory to Python path
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('.')");
        PyRun_SimpleString("sys.path.append('/usr/local/lib/python3.9/site-packages')");
        
        printf("Python bridge initialized - ALL Python libraries available!\n");
        python_initialized = 1;
    }
}

// Import any Python module
PyObject* aqualua_python_import(const char* module_name) {
    if (!python_initialized) {
        aqualua_init_python();
    }
    
    PyObject* module = PyImport_ImportModule(module_name);
    if (!module) {
        PyErr_Print();
        return NULL;
    }
    
    return module;
}

// Call any Python function
PyObject* aqualua_python_call(const char* module_name, const char* function_name, PyObject* args) {
    PyObject* module = aqualua_python_import(module_name);
    if (!module) return NULL;
    
    PyObject* func = PyObject_GetAttrString(module, function_name);
    if (!func || !PyCallable_Check(func)) {
        PyErr_Print();
        Py_DECREF(module);
        return NULL;
    }
    
    PyObject* result = PyObject_CallObject(func, args);
    if (!result) {
        PyErr_Print();
    }
    
    Py_DECREF(func);
    Py_DECREF(module);
    return result;
}

// Convert AquaLua tensor to NumPy array (zero-copy when possible)
PyObject* aqualua_tensor_to_numpy(void* tensor_data, int* shape, int ndim, int dtype) {
    PyObject* numpy = aqualua_python_import("numpy");
    if (!numpy) return NULL;
    
    // Create shape tuple
    PyObject* shape_tuple = PyTuple_New(ndim);
    for (int i = 0; i < ndim; i++) {
        PyTuple_SetItem(shape_tuple, i, PyLong_FromLong(shape[i]));
    }
    
    // Create NumPy array from memory buffer (zero-copy)
    PyObject* array_func = PyObject_GetAttrString(numpy, "frombuffer");
    PyObject* buffer = PyMemoryView_FromMemory((char*)tensor_data, 
                                              shape[0] * shape[1] * sizeof(float), 
                                              PyBUF_READ);
    
    PyObject* args = PyTuple_Pack(1, buffer);
    PyObject* numpy_array = PyObject_CallObject(array_func, args);
    
    // Reshape array
    PyObject* reshape_func = PyObject_GetAttrString(numpy_array, "reshape");
    PyObject* reshaped = PyObject_CallObject(reshape_func, PyTuple_Pack(1, shape_tuple));
    
    Py_DECREF(numpy);
    Py_DECREF(shape_tuple);
    Py_DECREF(array_func);
    Py_DECREF(buffer);
    Py_DECREF(args);
    Py_DECREF(numpy_array);
    Py_DECREF(reshape_func);
    
    return reshaped;
}

// Convert NumPy array back to AquaLua tensor
void* aqualua_numpy_to_tensor(PyObject* numpy_array, int** shape, int* ndim) {
    // Get array data pointer
    PyObject* data_ptr = PyObject_CallMethod(numpy_array, "__array_interface__", NULL);
    PyObject* data_dict = PyDict_GetItemString(data_ptr, "data");
    void* data = PyLong_AsVoidPtr(PyTuple_GetItem(data_dict, 0));
    
    // Get shape
    PyObject* shape_obj = PyObject_GetAttrString(numpy_array, "shape");
    *ndim = PyTuple_Size(shape_obj);
    *shape = malloc(*ndim * sizeof(int));
    
    for (int i = 0; i < *ndim; i++) {
        (*shape)[i] = PyLong_AsLong(PyTuple_GetItem(shape_obj, i));
    }
    
    Py_DECREF(data_ptr);
    Py_DECREF(shape_obj);
    
    return data;
}

// Execute arbitrary Python code
int aqualua_python_exec(const char* code) {
    if (!python_initialized) {
        aqualua_init_python();
    }
    
    int result = PyRun_SimpleString(code);
    if (result != 0) {
        PyErr_Print();
    }
    
    return result;
}

// Get Python object attribute
PyObject* aqualua_python_getattr(PyObject* obj, const char* attr_name) {
    return PyObject_GetAttrString(obj, attr_name);
}

// Set Python object attribute
int aqualua_python_setattr(PyObject* obj, const char* attr_name, PyObject* value) {
    return PyObject_SetAttrString(obj, attr_name, value);
}

// Create Python list from C array
PyObject* aqualua_create_python_list(void* data, int size, int dtype) {
    PyObject* list = PyList_New(size);
    
    if (dtype == 0) { // float
        float* float_data = (float*)data;
        for (int i = 0; i < size; i++) {
            PyList_SetItem(list, i, PyFloat_FromDouble(float_data[i]));
        }
    } else if (dtype == 1) { // int
        int* int_data = (int*)data;
        for (int i = 0; i < size; i++) {
            PyList_SetItem(list, i, PyLong_FromLong(int_data[i]));
        }
    }
    
    return list;
}

// Auto-install Python packages
int aqualua_install_package(const char* package_name) {
    char command[256];
    snprintf(command, sizeof(command), 
             "import subprocess; subprocess.check_call(['pip', 'install', '%s'])", 
             package_name);
    
    return aqualua_python_exec(command);
}

// Check if Python package is available
int aqualua_package_available(const char* package_name) {
    char code[256];
    snprintf(code, sizeof(code), 
             "try:\n    import %s\n    result = True\nexcept ImportError:\n    result = False", 
             package_name);
    
    PyRun_SimpleString(code);
    
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* result = PyObject_GetAttrString(main_module, "result");
    
    int available = PyObject_IsTrue(result);
    Py_DECREF(result);
    
    return available;
}

// Cleanup Python bridge
void aqualua_cleanup_python() {
    if (python_initialized) {
        Py_Finalize();
        python_initialized = 0;
        printf("Python bridge cleaned up\n");
    }
}