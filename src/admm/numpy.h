#include "Python.h"
#include <stdlib.h>
#include <string.h>

typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;

#define NPY_ARRAY_C_CONTIGUOUS    0x0001

/*
 * Set if array is a contiguous Fortran array: the first index varies
 * the fastest in memory (strides array is reverse of C-contiguous
 * array)
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_F_CONTIGUOUS    0x0002

/*
 * Note: all 0-d arrays are C_CONTIGUOUS and F_CONTIGUOUS. If a
 * 1-d array is C_CONTIGUOUS it is also F_CONTIGUOUS. Arrays with
 * more then one dimension can be C_CONTIGUOUS and F_CONTIGUOUS
 * at the same time if they have either zero or one element.
 * A higher dimensional array always has the same contiguity flags as
 * `array.squeeze()`; dimensions with `array.shape[dimension] == 1` are
 * effectively ignored when checking for contiguity.
 */

/*
 * If set, the array owns the data: it will be free'd when the array
 * is deleted.
 *
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_OWNDATA         0x0004

/*
 * An array never has the next four set; they're only used as parameter
 * flags to the various FromAny functions
 *
 * This flag may be requested in constructor functions.
 */

/* Cause a cast to occur regardless of whether or not it is safe. */
#define NPY_ARRAY_FORCECAST       0x0010

/*
 * Always copy the array. Returned arrays are always CONTIGUOUS,
 * ALIGNED, and WRITEABLE. See also: NPY_ARRAY_ENSURENOCOPY = 0x4000.
 *
 * This flag may be requested in constructor functions.
 */
#define NPY_ARRAY_ENSURECOPY      0x0020

/*
 * Make sure the returned array is a base-class ndarray
 *
 * This flag may be requested in constructor functions.
 */
#define NPY_ARRAY_ENSUREARRAY     0x0040

/*
 * Make sure that the strides are in units of the element size Needed
 * for some operations with record-arrays.
 *
 * This flag may be requested in constructor functions.
 */
#define NPY_ARRAY_ELEMENTSTRIDES  0x0080

/*
 * Array data is aligned on the appropriate memory address for the type
 * stored according to how the compiler would align things (e.g., an
 * array of integers (4 bytes each) starts on a memory address that's
 * a multiple of 4)
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_ALIGNED         0x0100

/*
 * Array data has the native endianness
 *
 * This flag may be requested in constructor functions.
 */
#define NPY_ARRAY_NOTSWAPPED      0x0200

/*
 * Array data is writeable
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_WRITEABLE       0x0400

/*
 * If this flag is set, then base contains a pointer to an array of
 * the same size that should be updated with the current contents of
 * this array when PyArray_ResolveWritebackIfCopy is called.
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_WRITEBACKIFCOPY 0x2000

/*
 * No copy may be made while converting from an object/array (result is a view)
 *
 * This flag may be requested in constructor functions.
 */
#define NPY_ARRAY_ENSURENOCOPY 0x4000






/* The item must be reference counted when it is inserted or extracted. */
#define NPY_ITEM_REFCOUNT   0x01
/* Same as needing REFCOUNT */
#define NPY_ITEM_HASOBJECT  0x01
/* Convert to list for pickling */
#define NPY_LIST_PICKLE     0x02
/* The item is a POINTER  */
#define NPY_ITEM_IS_POINTER 0x04
/* memory needs to be initialized for this data-type */
#define NPY_NEEDS_INIT      0x08
/* operations need Python C-API so don't give-up thread. */
#define NPY_NEEDS_PYAPI     0x10
/* Use f.getitem when extracting elements of this data-type */
#define NPY_USE_GETITEM     0x20
/* Use f.setitem when setting creating 0-d array from this data-type.*/
#define NPY_USE_SETITEM     0x40
/* A sticky flag specifically for structured arrays */
#define NPY_ALIGNED_STRUCT  0x80

typedef PyObject* (*dmatmul_kernel)(void* l, void* r, int row, int col);

typedef struct {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    /*
    PyArray_ArrayDescr *subarray;
    PyObject *fields;
    PyObject *names;
    PyArray_ArrFuncs *f;
    PyObject *metadata;
    NpyAuxData *c_metadata;
    npy_hash_t hash;
    */
} __PyArray_Descr;

typedef struct ___PyArrayObject {
    PyObject_HEAD
    char *data;
    int nd;
    npy_intp *dimensions;
    npy_intp *strides;
    PyObject *base;
    __PyArray_Descr *descr;
    int flags;
    PyObject *weakreflist;
    /* version dependent private members */
} __PyArrayObject;

void numpy_ensure_c_order(void* arr);

int numpy_ndim(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->nd;
}

void numpy_shape(void* arr, uint32_t* shape)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    int i;
    for (i = 0; i < obj->nd; i++)
    {
        npy_intp d = obj->dimensions[i];
        shape[i] = (uint32_t)d;
    }
}

void numpy_strides(void* arr, uint32_t* strides)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    int i;
    for (i = 0; i < obj->nd; i++)
    {
        npy_intp d = obj->strides[i];
        strides[i] = (uint32_t)d;
    }
}

int numpy_c_contiguous(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->flags & NPY_ARRAY_C_CONTIGUOUS;
}

int numpy_f_contiguous(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->flags & NPY_ARRAY_F_CONTIGUOUS;
}

int numpy_aligned(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->flags & NPY_ARRAY_ALIGNED;
}

int numpy_elestride(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->flags & NPY_ARRAY_ELEMENTSTRIDES;
}

int numpy_elepointer(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->descr->flags & NPY_ITEM_IS_POINTER;
}

int numpy_elesize(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->descr->elsize;
}

int numpy_alignment(void* arr)
{
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->descr->alignment; 
}

int numpy_size(void* arr)
{
    int i;
    int size = 1;
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    for (i = 0; i < obj->nd; i++)
    {
        size *= obj->dimensions[i];
    }
    return size;
}

void* numpy_get(void* arr, uint32_t* idx)
{
    int i;
    uint32_t offset = 0;
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    for (i = 0; i < obj->nd; i++)
    {
        offset += idx[i] * obj->strides[i];
    }
    return obj->data + offset;
}

PyObject* numpy_getobj(void* arr, uint32_t* idx)
{
    PyObject** addr = (PyObject**)numpy_get(arr, idx);
    PyObject* obj = *addr;
    Py_INCREF(obj);
    return obj;
}

void numpy_set_obj_and_incref(void* arr, uint32_t* idx, void* obj)
{
    int i;
    uint32_t offset = 0;
    __PyArrayObject* arrobj = (__PyArrayObject*)arr;
    for (i = 0; i < arrobj->nd; i++)
    {
        offset += idx[i] * arrobj->strides[i];
    }
    ((void**)(arrobj->data + offset))[0] = obj;
    Py_INCREF(obj);
}

void* numpy_buffer(void* arr)
{
    numpy_ensure_c_order(arr);
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return obj->data;
}

char* numpy_char_buffer(void* arr)
{
    numpy_ensure_c_order(arr);
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return (char*)(obj->data);
}

int* numpy_int_buffer(void* arr)
{
    numpy_ensure_c_order(arr);
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return (int*)(obj->data);
}

unsigned long* numpy_ulong_buffer(void* arr)
{
    numpy_ensure_c_order(arr);
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return (unsigned long*)(obj->data);
}

double* numpy_double_buffer(void* arr)
{
    numpy_ensure_c_order(arr);
    __PyArrayObject* obj = (__PyArrayObject*)arr;
    return (double*)(obj->data);
}

void numpy_ensure_c_order(void* arr)
{
    int64_t i;
    size_t j;
    size_t t;
    size_t r;
    char* d;
    size_t size = 1;
    size_t* w1 = NULL;
    size_t* w2 = NULL;
    int elesize;
    char* buffer;

    __PyArrayObject* obj = (__PyArrayObject*)arr;
    d = (char*)obj->data;
    elesize = obj->descr->elsize;

    if ((obj->flags & NPY_ARRAY_C_CONTIGUOUS) == 0)
    {
        if (obj->nd > 0)
        {
            w1 = (size_t*) malloc(sizeof(size_t) * obj->nd);
            w1[obj->nd - 1] = 1;
        }
        if (obj->nd > 0)
        {
            w2 = (size_t*) malloc(sizeof(size_t) * obj->nd);
            w2[obj->nd - 1] = 1;
        }

        for (i = obj->nd - 2; i >= 0; i--)
        {
            w1[i] = w1[i + 1] * obj->dimensions[i + 1];
            w2[i] = w2[i + 1] * obj->dimensions[obj->nd - i - 2];
        }

        for (i = 0; i < obj->nd; i++)
            size *= obj->dimensions[i];

        buffer = (char*)malloc(size * elesize);
        
        // Calculate strides
        for (i = 0; i < obj->nd; i++)
            obj->strides[i] = w1[i] * elesize;

        // Transpose
        for (i = 0; i < (int64_t)size; i++)
        {
            t = 0;
            r = i;
            for (j = 0; j < (size_t)obj->nd; j++)
            {
                t += (r / w2[j]) * w1[obj->nd - j - 1];
                r = r % w2[j];
            }

            memcpy(buffer + t * elesize, d + i * elesize, elesize);
        }

        free(obj->data);
        obj->data = buffer;
        if (obj->nd > 0) free(w1);
        if (obj->nd > 0) free(w2);

        obj->flags &= ~NPY_ARRAY_F_CONTIGUOUS;
        obj->flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
}


void numpy_copy_arr_to_ptr(void* arr, double* ptr)
{
    size_t size = numpy_size(arr);
    for (size_t i = 0; i < size; i++)
    memcpy(ptr, numpy_double_buffer(arr), size * sizeof(double));
}

void numpy_copy_ptr_to_arr(double* ptr, void* arr)
{
    size_t size = numpy_size(arr);
    memcpy(numpy_double_buffer(arr), ptr, size * sizeof(double));
}