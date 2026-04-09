

cdef class UDFBase(TensorLike):
    '''@APIDOC(py.UDFBase)'''
    cdef int index_
    cdef int type_

    @constant
    def index(self):
        return self.index_

    @constant
    def type(self):
        return AT_EXPR

    @constant
    def shape(self):
        # Check if it's illegal as early as possible
        # Method `shape` will be called for each time it participates arithmetic operation
        getUdfId(self)
        return tuple()

    def arguments(self):
        '''@APIDOC(py.UDFBase.arguments)'''
        raise NotImplementedError

    def eval(self, tensorlist):
        '''@APIDOC(py.UDFBase.eval)'''
        raise NotImplementedError

    def argmin(self, lamb, tensorlist):
        '''@APIDOC(py.UDFBase.argmin)'''
        raise NotImplementedError

    def grad(self, tensorlist):
        '''Compute gradient of the UDF at the given point.'''
        raise NotImplementedError

cdef hasMethod(obj, fnames):
    for fun in fnames:
        if not hasattr(obj, fun) or not callable(getattr(obj, fun)):
            return False
        method = getattr(obj, fun)
        className = method.__qualname__[:-len(fun) - 1]
        if className == UDFBase.__qualname__:
            return False
    return True

cdef assertUdf(obj):
    has_argmin = hasMethod(obj, ("arguments", "eval", "argmin"))
    has_grad = hasMethod(obj, ("arguments", "eval", "grad"))
    if not isinstance(obj, UDFBase) or not (has_argmin or has_grad):
        raise TypeError("not a valid UDF: must implement (arguments, eval, argmin) or (arguments, eval, grad)")

cdef int _udf_eval(void* userdata, int nexprs, int* ndims, int* shapes, double* tensors, double* data):
    cdef object udf
    cdef size_t dimoffset
    cdef size_t dataoffset

    dimoffset = 0
    dataoffset = 0

    try:
        udf = <object>userdata
        arrays = []
        for i in range(nexprs):
            shapelist = []
            for j in range(ndims[i]):
                shapelist.append(shapes[dimoffset])
                dimoffset += 1

            ndarr = _np.empty(tuple(shapelist), dtype=_np.float64)
            numpy_copy_ptr_to_arr(tensors + dataoffset, ndarr)
            dataoffset += _np.prod(ndarr.shape)
            arrays.append(ndarr)

        result = udf.eval(arrays)

        if isinstance(result, _np.ndarray) and _np.prod(result.shape) != 1:
            raise ValueError("{}.eval is expected to return a scalar value".format(type(udf).__name__))

        if isinstance(result, _np.ndarray):
            result = result.flat[0]
        
        if result is None or not isinstance(result, _numbers.Number):
            raise TypeError("{}.eval is expected to return a numeric value".format(type(udf).__name__))

        data[0] = float(result)
    except BaseException as e:
        _traceback.print_exc()
        return -1

    return 0

cdef int _udf_argmin(void* userdata, double lam, int nexprs, int* ndims, int* shapes, double* tensors):
    cdef object udf
    cdef size_t dimoffset
    cdef size_t dataoffset

    dimoffset = 0
    dataoffset = 0

    try:
        udf = <object>userdata
        arrays = []

        for i in range(nexprs):
            shapelist = []
            for j in range(ndims[i]):
                shapelist.append(shapes[dimoffset])
                dimoffset += 1
            ndarr = _np.empty(tuple(shapelist), dtype=_np.float64)
            numpy_copy_ptr_to_arr(tensors + dataoffset, ndarr)
            dataoffset += _np.prod(ndarr.shape)
            arrays.append(ndarr)

        result = udf.argmin(lam, arrays)

        if result is None:
            raise StopIteration("None value returned")

        if isinstance(result, (_numbers.Number, _np.ndarray, _sp.spmatrix, Constant)):
            result = [result]

        if not isinstance(result, (tuple, list)):
            raise TypeError("{}.argmin is expected to return a list".format(type(udf).__name__))
        
        if len(result) != nexprs:
            raise ValueError("{}.argmin is expected to return a list with {} elements".format(type(udf).__name__, nexprs))

        dataoffset = 0
        for i in range(len(result)):
            if isinstance(result[i], _numbers.Number):
                result[i] = _np.array(result[i], dtype=_np.float64)
            elif isinstance(result[i], _sp.spmatrix):
                result[i] = result[i].toarray().astype(dtype=_np.float64)
            elif isinstance(result[i], Constant):
                result[i] = result[i].asDense().data

            if not isinstance(result[i], _np.ndarray):
                result[i] = _np.array(result[i], dtype=_np.float64)

            if _np.prod(result[i].shape) == 1 and _np.prod(arrays[i].shape) == 1:
                pass
            elif result[i].shape != arrays[i].shape:
                raise ValueError("{}.argmin returned a tensor with mismatched shape".format(type(udf).__name__))

            numpy_copy_arr_to_ptr(result[i], tensors + dataoffset);
            dataoffset += _np.prod(result[i].shape)

    except BaseException as e:
        _traceback.print_exc()
        return -1

    return 0

cdef int _udf_grad(void* userdata, int nexprs, int* ndims, int* shapes, double* tensors, double* grad_out):
    cdef object udf
    cdef size_t dimoffset
    cdef size_t dataoffset

    dimoffset = 0
    dataoffset = 0

    try:
        udf = <object>userdata
        arrays = []
        for i in range(nexprs):
            shapelist = []
            for j in range(ndims[i]):
                shapelist.append(shapes[dimoffset])
                dimoffset += 1

            ndarr = _np.empty(tuple(shapelist), dtype=_np.float64)
            numpy_copy_ptr_to_arr(tensors + dataoffset, ndarr)
            dataoffset += _np.prod(ndarr.shape)
            arrays.append(ndarr)

        result = udf.grad(arrays)

        if result is None:
            raise StopIteration("None value returned from grad")

        if isinstance(result, (_numbers.Number, _np.ndarray, _sp.spmatrix, Constant)):
            result = [result]

        if not isinstance(result, (tuple, list)):
            raise TypeError("{}.grad is expected to return a list".format(type(udf).__name__))

        if len(result) != nexprs:
            raise ValueError("{}.grad is expected to return a list with {} elements".format(type(udf).__name__, nexprs))

        dataoffset = 0
        for i in range(len(result)):
            if isinstance(result[i], _numbers.Number):
                result[i] = _np.array(result[i], dtype=_np.float64)
            elif isinstance(result[i], _sp.spmatrix):
                result[i] = result[i].toarray().astype(dtype=_np.float64)
            elif isinstance(result[i], Constant):
                result[i] = result[i].asDense().data

            if not isinstance(result[i], _np.ndarray):
                result[i] = _np.array(result[i], dtype=_np.float64)

            if _np.prod(result[i].shape) == 1 and _np.prod(arrays[i].shape) == 1:
                pass
            elif result[i].shape != arrays[i].shape:
                raise ValueError("{}.grad returned a tensor with mismatched shape".format(type(udf).__name__))

            numpy_copy_arr_to_ptr(result[i], grad_out + dataoffset);
            dataoffset += _np.prod(result[i].shape)

    except BaseException as e:
        _traceback.print_exc()
        return -1

    return 0

cdef int udf_eval(void* userdata, int nexprs, int* ndims, int* shapes, double* tensors, double* data):
    gillock = pygillock()

    try:
        # Never inline the following function call
        # Otherwise python may raise error 'GIL released'
        return _udf_eval(userdata, nexprs, ndims, shapes, tensors, data)
    finally:
        pygilrelease(gillock)

cdef int udf_argmin(void* userdata, double lam, int nexprs, int* ndims, int* shapes, double* tensors):
    gillock = pygillock()

    try:
        # Never inline the following function call
        # Otherwise python may raise error 'GIL released'
        return _udf_argmin(userdata, lam, nexprs, ndims, shapes, tensors)
    finally:
        pygilrelease(gillock)

cdef int udf_grad(void* userdata, int nexprs, int* ndims, int* shapes, double* tensors, double* grad_out):
    gillock = pygillock()

    try:
        return _udf_grad(userdata, nexprs, ndims, shapes, tensors, grad_out)
    finally:
        pygilrelease(gillock)

# udf class type -> udfid
cdef dict udfs = {}

cdef getUdfId(obj):
    cdef int udfid
    assertUdf(obj)
    udft = type(obj)

    if udft in udfs:
        return udfs[udft]

    name = udft.__qualname__
    rawname = bytes(name, "utf8")

    has_argmin = hasMethod(obj, ("argmin",))
    has_grad = hasMethod(obj, ("grad",))

    if has_argmin:
        err = admm_custom_udf(rawname, <value_eval_t>udf_eval, <argmin_t>udf_argmin, &udfid)
    elif has_grad:
        err = admm_custom_udf_with_grad(rawname, <value_eval_t>udf_eval, <grad_t>udf_grad, &udfid)
    else:
        raise TypeError("UDF must implement argmin() or grad()")

    if err != 0: raise ADMMError(err)

    udfs[udft] = udfid
    return udfid
