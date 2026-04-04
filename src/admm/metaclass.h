#include <stdlib.h>

/* All args flags of a PyMethod */
#define METH_ALLARGS (METH_VARARGS | METH_KEYWORDS | METH_NOARGS | METH_O)

/* Compatibility macro for Py_SET_TYPE */
#if !defined(Py_SET_TYPE)
#if PY_VERSION_HEX < 0x030900A4
static inline void set_object_type(PyObject* obj, PyTypeObject* tp)
{
    obj->ob_type = tp;
}
#define Py_SET_TYPE(obj, tp) set_object_type((PyObject*)(obj), tp)
#endif
#endif

static PyObject* triple_none_cache = NULL;

static CYTHON_INLINE PyObject* invoke_method_descriptor(PyMethodDescrObject* descriptor, const char* type_name, PyObject* instance)
{
    PyObject* callable = PyCFunction_NewEx(descriptor->d_method, instance, NULL);
    if (callable == NULL) {
        return NULL;
    }
    
    PyObject* empty_args = PyTuple_New(0);
    PyObject* retval = PyObject_CallObject(callable, empty_args);
    
    Py_DECREF(empty_args);
    Py_DECREF(callable);
    
    return retval;
}

static CYTHON_INLINE int Mindopt_PyType_Ready(PyTypeObject* type_obj)
{
    int status = PyType_Ready(type_obj);
    if (status < 0) {
        return status;
    }

    PyObject* metaclass_getter = PyObject_GetAttrString((PyObject*)type_obj, "__metacls__");
    PyTypeObject* meta_type;
    
    if (metaclass_getter != NULL) {
        meta_type = (PyTypeObject*)invoke_method_descriptor(
            (PyMethodDescrObject*)metaclass_getter, 
            type_obj->tp_name, 
            Py_None
        );
        Py_DECREF(metaclass_getter);
        
        if (meta_type == NULL) {
            return -1;
        }

        if (!PyType_Check(meta_type)) {
            PyErr_SetString(PyExc_TypeError, "__metacls__ did not return a type");
            return -1;
        }

        Py_SET_TYPE(type_obj, meta_type);
        PyType_Modified(type_obj);
    } else {
        PyErr_Clear();
        meta_type = Py_TYPE(type_obj);
    }

    initproc initializer = meta_type->tp_init;
    if (initializer == NULL) {
        return 0;
    }
    if (initializer == PyType_Type.tp_init) {
        return 0;
    }

    if (meta_type->tp_basicsize != PyType_Type.tp_basicsize) {
        PyErr_SetString(
            PyExc_TypeError,
            "metaclass is not compatible with 'type' (you cannot use cdef attributes in Cython metaclasses)"
        );
        return -1;
    }

    if (triple_none_cache == NULL) {
        triple_none_cache = PyTuple_Pack(3, Py_None, Py_None, Py_None);
        if (triple_none_cache == NULL) {
            return -1;
        }
    }
    
    return initializer((PyObject*)type_obj, triple_none_cache, NULL);
}

#define PyType_Ready(type_obj) Mindopt_PyType_Ready(type_obj)