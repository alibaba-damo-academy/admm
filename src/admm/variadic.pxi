def hstack(*tensors):
    '''@APIDOC(py.hstack)'''
    cdef tensor_t* tarr
    cdef int rdim
    cdef int* ndim
    cdef int** shape

    ID = 901

    nopr = len(tensors)
    t = [ensure_array_type(tensor) for tensor in tensors]
    arr = [_np.array(tensor.shape, dtype=_np.int32) for tensor in t]
    maxdim = _np.max([tensor.ndim for tensor in t])
    # shape_eval may promote dimensions (e.g. 1-D → 2-D), so allocate extra room
    _bufdim = maxdim + 1 if maxdim + 1 > 2 else 2
    rshapearr = _np.empty(_bufdim, dtype=_np.int32)

    ndim = <int*>malloc(sizeof(int) * nopr)
    shape = <int**>malloc(sizeof(int*) * nopr)

    try:
        for i in range(len(t)):
            ndim[i] = t[i].ndim
            shape[i] = numpy_int_buffer(arr[i])

        err = admm_shape_variadic_fun(ID, nopr, ndim, shape, &rdim, numpy_int_buffer(rshapearr))
        if err != 0: raise ADMMError(err)
    finally:
        free(ndim)
        free(shape)

    def merge(t, start, stop):
        numConst = stop - start
        constants = [Constant(t[i]) for i in range(start, stop)]
        tarr = <tensor_t*>malloc(sizeof(tensor_t) * numConst)
        try:
            for i in range(numConst):
                tarr[i] = (<Constant>(constants[i])).tensor_
            err = admm_tensor_variadic_fun(ID, numConst, tarr)
            if err != 0: raise ADMMError(err)
            return constants[0]
        finally:
            free(tarr)

    merged = []
    constStart = 0

    for i in range(len(t)):
        if not isConstExp(t[i]):
            numConst = i - constStart
            if numConst == 1:
                merged.append(t[constStart])
            elif numConst > 1:
                merged.append(merge(t, constStart, i))

            constStart = i + 1
            merged.append(t[i])

    numConst = len(t) - constStart
    if numConst == 1:
        merged.append(t[constStart])
    elif numConst > 1:
        merged.append(merge(t, constStart, len(t)))

    if len(merged) == 1:
        return merged[0]

    return Expr(tuple(rshapearr[:rdim].tolist()), ID, merged)


def vstack(*tensors):
    '''@APIDOC(py.vstack)'''
    cdef tensor_t* tarr
    cdef int rdim
    cdef int* ndim
    cdef int** shape

    ID = 902

    nopr = len(tensors)
    t = [ensure_array_type(tensor) for tensor in tensors]
    arr = [_np.array(tensor.shape, dtype=_np.int32) for tensor in t]
    maxdim = _np.max([tensor.ndim for tensor in t])
    # shape_eval may promote dimensions (e.g. 1-D → 2-D), so allocate extra room
    _bufdim = maxdim + 1 if maxdim + 1 > 2 else 2
    rshapearr = _np.empty(_bufdim, dtype=_np.int32)

    ndim = <int*>malloc(sizeof(int) * nopr)
    shape = <int**>malloc(sizeof(int*) * nopr)

    try:
        for i in range(len(t)):
            ndim[i] = t[i].ndim
            shape[i] = numpy_int_buffer(arr[i])

        err = admm_shape_variadic_fun(ID, nopr, ndim, shape, &rdim, numpy_int_buffer(rshapearr))
        if err != 0: raise ADMMError(err)
    finally:
        free(ndim)
        free(shape)

    def merge(t, start, stop):
        numConst = stop - start
        constants = [Constant(t[i]) for i in range(start, stop)]
        tarr = <tensor_t*>malloc(sizeof(tensor_t) * numConst)
        try:
            for i in range(numConst):
                tarr[i] = (<Constant>(constants[i])).tensor_
            err = admm_tensor_variadic_fun(ID, numConst, tarr)
            if err != 0: raise ADMMError(err)
            return constants[0]
        finally:
            free(tarr)

    merged = []
    constStart = 0

    for i in range(len(t)):
        if not isConstExp(t[i]):
            numConst = i - constStart
            if numConst == 1:
                merged.append(t[constStart])
            elif numConst > 1:
                merged.append(merge(t, constStart, i))

            constStart = i + 1
            merged.append(t[i])

    numConst = len(t) - constStart
    if numConst == 1:
        merged.append(t[constStart])
    elif numConst > 1:
        merged.append(merge(t, constStart, len(t)))

    if len(merged) == 1:
        return merged[0]

    return Expr(tuple(rshapearr[:rdim].tolist()), ID, merged)