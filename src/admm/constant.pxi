cdef class Constant(TensorLike):
    '''@APIDOC(py.Constant)'''
    cdef tensor_t tensor_
    cdef int freed_
    cdef Model model_
    cdef int id_

    def __init__(self, arr, ind=None, val=None):
        '''@APIDOC(py.Constant.__init__)'''
        self.freed_ = 1
        self.id_ = -1
        self.model_ = None

        if isinstance(arr, Constant):
            err = admm_tensor_copy(&self.tensor_, (<Constant>arr).tensor_)
            if err != 0: raise ADMMError(err)
            self.freed_ = 0
            
        elif ind is None and val is None:
            arr = ensure_array_type(arr)
            shape = _np.array(arr.shape, dtype=_np.int32)

            if isinstance(arr, _sp.spmatrix):
                arr = arr.tocoo()
                if arr.nnz == 0:
                    # Empty sparse matrix: create as dense zeros
                    arr = _np.zeros(shape, dtype=_np.float64)
                    err = admm_create_dense(&self.tensor_, shape.size, numpy_int_buffer(shape), numpy_double_buffer(arr))
                else:
                    ind = arr.row * shape[1] + arr.col
                    ind = _np.ascontiguousarray(ind, dtype=_np.uint64)
                    val = _np.ascontiguousarray(arr.data, dtype=_np.float64)
                    err = admm_create_sparse(&self.tensor_, shape.size, numpy_int_buffer(shape),
                        arr.nnz, <size_t*>numpy_ulong_buffer(ind), numpy_double_buffer(val))
                if err != 0: raise ADMMError(err)
                self.freed_ = 0
            else:
                arr = _np.ascontiguousarray(arr, dtype=_np.float64)
                err = admm_create_dense(&self.tensor_, shape.size, numpy_int_buffer(shape), numpy_double_buffer(arr))
                if err != 0: raise ADMMError(err)
                self.freed_ = 0
        else:
            if arr is None or ind is None or val is None:
                raise TypeError("expect 3 arrays, shape, index, value")

            shape = ensure_array_type(arr)
            val = ensure_array_type(val)

            if len(ind) != len(val):
                raise ValueError("index has different size from value")

            if len(ind) > 0:
                if isinstance(ind[0], _numbers.Number):
                    ind = ensure_array_type(ind)
                else:
                    wts = [1]
                    for i in range(shape.size - 1): wts.append(wts[-1] * shape[-i - 1])
                    wts.reverse()
                    try:
                        ind = _np.matmul(ind, wts)
                    except:
                        raise ValueError("index has different number of dimension from shape")

            if not isinstance(shape, _np.ndarray) or not isinstance(ind, _np.ndarray) or not isinstance(val, _np.ndarray):
                raise TypeError("expect 3 arrays, shape, index, value")

            shape = _np.array(shape, dtype=_np.int32)
            ind = _np.ascontiguousarray(ind, dtype=_np.uint64)
            val = _np.ascontiguousarray(val, dtype=_np.float64)

            space = _np.prod(shape, dtype=_np.int64)
            if _np.any(ind >= space): raise IndexError("index is out of range")

            err = admm_create_sparse(&self.tensor_, shape.size, numpy_int_buffer(shape),
                    ind.size, <size_t*>numpy_ulong_buffer(ind), numpy_double_buffer(val))

            if err != 0: raise ADMMError(err)
            self.freed_ = 0

    cdef saved(self, Model model, int isdense, int cid):
        cdef int ndim
        cdef int nnz

        self.model_ = model
        self.id_ = cid
        self.dispose()

        err = admm_create_from_model(&self.tensor_, model.mdl_, isdense, cid)
        if err != 0: raise ADMMError(err)
        self.freed_ = 0
        

    @constant
    def index(self):
        '''@APIDOC(py.TensorLike.index)'''
        return self.id_

    @constant
    def type(self):
        '''@APIDOC(py.TensorLike.type)'''
        return AT_DENSE if self.isDense() else AT_SPARSE

    def isDense(self):
        '''@APIDOC(py.Constant.isDense)'''
        dense = admm_tensor_is_dense(self.tensor_)
        return dense != 0

    def isSparse(self):
        '''@APIDOC(py.Constant.isSparse)'''
        return not self.isDense()

    @constant
    def data(self):
        '''@APIDOC(py.Constant.data)'''
        cdef size_t nzs

        if self.isDense():
            data = _np.empty(self.shape, dtype=_np.float64)
            err = admm_tensor_get_dense(self.tensor_, numpy_double_buffer(data));
            if err != 0: raise ADMMError(err)
            return data
        else:
            err = admm_tensor_get_sparse(self.tensor_, &nzs, NULL, NULL)
            if err != 0: raise ADMMError(err)

            ind = _np.empty(nzs, dtype=_np.uint64)
            val = _np.empty(nzs, dtype=_np.float64)

            err = admm_tensor_get_sparse(self.tensor_, &nzs, <size_t*>numpy_ulong_buffer(ind), numpy_double_buffer(val))
            if err != 0: raise ADMMError(err)
            return (ind, val)

    @constant
    def ndim(self):
        '''@APIDOC(py.TensorLike.ndim)'''
        return len(self.shape)

    @constant
    def shape(self):
        '''@APIDOC(py.TensorLike.shape)'''
        cdef int ndim

        err = admm_tensor_get_shape(self.tensor_, &ndim, NULL)
        if err != 0: raise ADMMError(err)

        shape = _np.empty(ndim, dtype=_np.int32)

        err = admm_tensor_get_shape(self.tensor_, &ndim, numpy_int_buffer(shape))
        if err != 0: raise ADMMError(err)
        return tuple(shape.tolist())


    def asScalar(self):
        '''@APIDOC(py.Constant.asScalar)'''
        shape = self.shape
        if _np.prod(shape) != 1:
            raise ValueError("not a scalar")
        if self.isDense():
            return self.data.flat[0]
        else: return self.data[1].flat[0]

    def asDense(self):
        '''@APIDOC(py.Constant.asDense)'''
        if self.isDense():
            return self
        else:
            res = Constant(self)
            err = admm_tensor_toggle_type((<Constant>res).tensor_)
            if err != 0: raise ADMMError(err)
            return res

    def asSparse(self):
        '''@APIDOC(py.Constant.asSparse)'''
        if self.isSparse():
            return self
        else:
            res = Constant(self)
            err = admm_tensor_toggle_type((<Constant>res).tensor_)
            if err != 0: raise ADMMError(err)
            return res

    def hasAttr(self, **kws):
        '''@APIDOC(py.Constant.hasAttr)'''
        cdef int attr
        cdef int res

        attr = 0

        for key in kws:
            if key not in tensorattrnames:
                raise KeyError("unknown attr {}".format(key))
            if kws[key]:
                attr |= (1 << tensorattrnames[key])

        err = admm_tensor_has_attr(self.tensor_, attr, &res)
        if err != 0: raise ADMMError(err)
        return res != 0

    @constant
    def T(self):
        '''@APIDOC(py.Constant.T)'''
        res = Constant(self)
        err = admm_tensor_unary_op((<Constant>res).tensor_, OT_T)
        if err != 0: raise ADMMError(err)
        return res

    def reshape(self, *shape):
        '''@APIDOC(py.Constant.reshape)'''
        res = Constant(self)
        shapearr = _np.array(shape, dtype=_np.int32)
        err = admm_tensor_reshape_op((<Constant>res).tensor_, shapearr.size, numpy_int_buffer(shapearr))
        if err != 0: raise ADMMError(err)
        return res

    def __neg__(self):
        '''@APIDOC(py.Constant.__neg__)'''
        res = Constant(self)
        err = admm_tensor_unary_op((<Constant>res).tensor_, OT_NEG)
        if err != 0: raise ADMMError(err)
        return res

    def __getitem__(self, slc):
        '''@APIDOC(py.Constant.__getitem__)'''
        res = Constant(0)

        nranges = 1 if isinstance(slc, (_numbers.Number, slice)) else len(slc)

        shape = self.shape

        starts = _np.empty((nranges, ), dtype=_np.int32)
        stops = _np.empty((nranges, ), dtype=_np.int32)
        steps = _np.empty((nranges, ), dtype=_np.int32)

        if nranges == 1: slc = [slc]

        for i in range(nranges):
            if isinstance(slc[i], _numbers.Number):
                starts[i], stops[i], steps[i] = slc[i], -1, 1
            else:
                triple = slc[i].indices(self.shape[i])
                starts[i], stops[i], steps[i] = triple
            if -starts[i] > shape[i] or starts[i] >= shape[i]:
                raise IndexError("index is out of range")
            if starts[i] < 0: starts[i] = shape[i] + starts[i]

        err = admm_tensor_slice_op(self.tensor_, nranges, numpy_int_buffer(starts), 
                numpy_int_buffer(stops), numpy_int_buffer(steps), (<Constant>res).tensor_)

        if err != 0: raise ADMMError(err)
        return res

    cdef _binop(self, l, r, op):
        # Invoke super op
        if not isinstance(l, Constant) and isinstance(l, TensorLike):
            return None
        if not isinstance(r, Constant) and isinstance(r, TensorLike):
            return None

        # First operand l expected to be a copy
        l = Constant(l)
        # Second operand r expected to be Constant typed
        if not isinstance(r, Constant):
            r = Constant(r)        

        err = admm_tensor_binary_op((<Constant>l).tensor_, (<Constant>r).tensor_, op)
        if err != 0: raise ADMMError(err)
        return l

    def __add__(self, r):
        '''@APIDOC(py.Constant.__add__)'''
        res = self._binop(self, r, OT_ADD)
        return res if res is not None else super().__add__(r)

    def __radd__(self, l):
        '''@APIDOC(py.Constant.__radd__)'''
        res = self._binop(l, self, OT_ADD)
        return res if res is not None else super().__radd__(l)

    def __sub__(self, r):
        '''@APIDOC(py.Constant.__sub__)'''
        res = self._binop(self, r, OT_SUB)
        return res if res is not None else super().__sub__(r)

    def __rsub__(self, l):
        '''@APIDOC(py.Constant.__rsub__)'''
        res = self._binop(l, self, OT_SUB)
        return res if res is not None else super().__rsub__(l)

    def __mul__(self, r):
        '''@APIDOC(py.Constant.__mul__)'''
        res = self._binop(self, r, OT_MUL)
        return res if res is not None else super().__mul__(r)

    def __rmul__(self, l):
        '''@APIDOC(py.Constant.__rmul__)'''
        res = self._binop(l, self, OT_MUL)
        return res if res is not None else super().__rmul__(l)

    def __truediv__(self, r):
        '''@APIDOC(py.Constant.__mul__)'''
        res = self._binop(self, r, OT_DIV)
        return res if res is not None else super().__truediv__(r)

    def __rtruediv__(self, l):
        '''@APIDOC(py.Constant.__rmul__)'''
        res = self._binop(l, self, OT_DIV)
        return res if res is not None else super().__rtruediv__(l)

    def __matmul__(self, r):
        '''@APIDOC(py.Constant.__matmul__)'''
        res = self._binop(self, r, OT_MML)
        return res if res is not None else super().__matmul__(r)

    def __rmatmul__(self, l):
        '''@APIDOC(py.Constant.__rmatmul__)'''
        res = self._binop(l, self, OT_MML)
        return res if res is not None else super().__rmatmul__(l)
    
    def __pow__(self, r):
        '''@APIDOC(py.Constant.__pow__)'''
        res = self._binop(self, r, OT_POW)
        return res if res is not None else super().__pow__(r)

    def __rpow__(self, l):
        '''@APIDOC(py.Constant.__rpow__)'''
        res = self._binop(l, self, OT_POW)
        return res if res is not None else super().__rpow__(l)

    def dispose(self):
        '''@APIDOC(py.Constant.dispose)'''
        if self.freed_ == 0:
            self.freed_ = 1
            admm_destroy_tensor(self.tensor_)

    def __str__(self):
        cdef size_t nzs
        shape = self.shape

        if self.isDense():
            return self.data.__str__()
        else:
            ind, val = self.data
            
            wts = [1]
            for i in range(len(shape)): wts.append(wts[-1] * shape[-i - 1])
            wts.pop()

            result = str(shape)

            minval = inf
            maxval = -1
            hasminus = False

            fmt = "{:.8}"

            r = range(val.size)
            if val.size > 32:
                r = list(range(16))
                r += list(range(val.size - 16, val.size))
            for i in r:
                v = _builtins.abs(val[i])
                minval = _builtins.min(minval, v)
                maxval = _builtins.max(maxval, v)

                if 1000 * (_builtins.max(minval, 1)) <= maxval:
                    fmt = "{:.8e}"
                if str(val[i]).startswith("-"):
                    hasminus = True

            for i in r:
                cord = []
                index = ind[i]
                for j in range(len(shape)):
                    cord.append(int(index // wts[-j - 1]))
                    index %= wts[-j - 1]
                result += "\n"
                value = fmt.format(val[i])
                if not value.startswith('-') and hasminus: value = " " + value
                result += str(tuple(cord)) + " = " + value

                if isinstance(r, list) and i == 15:
                    result += "\n... {} nonzero(s) omitted ...".format(val.size - 32)

            return result


    def __repr__(self):
        if self.isDense():
            return "Dense({}\n{})".format(self.shape, self.__str__())
        return "Sparse({})".format(self.__str__())

    def __del__(self):
        self.dispose()

    def __dealloc__(self):
        self.dispose()

    def __array__(self, dtype = None):
        arr = self if self.isDense() else self.asDense()
        data = arr.data
        return data if dtype is None else data.astype(dtype)
