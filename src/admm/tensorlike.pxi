
cdef class TensorLike:
    '''@APIDOC(py.TensorLike)'''

    @constant
    def ndim(self):
        '''@APIDOC(py.TensorLike.ndim)'''
        return len(self.shape)

    @constant
    def shape(self):
        '''@APIDOC(py.TensorLike.shape)'''
        raise NotImplementedError()

    @constant
    def index(self):
        '''@APIDOC(py.TensorLike.index)'''
        raise NotImplementedError()

    @constant
    def type(self):
        '''@APIDOC(py.TensorLike.type)'''
        raise NotImplementedError()

    @constant
    def T(self):
        '''@APIDOC(py.TensorLike.T)'''
        cdef int rdim
        shapearr = _np.array(self.shape, dtype=_np.int32)
        rshapearr = _np.empty_like(shapearr)
        err = admm_shape_t(shapearr.size, numpy_int_buffer(shapearr), &rdim, numpy_int_buffer(rshapearr))

        if err != 0: raise ADMMError(err)
        return Expr(tuple(rshapearr.tolist()[:rdim]), OT_T, [self])

    def reshape(self, *shape):
        '''@APIDOC(py.TensorLike.reshape)'''
        oldshapearr = _np.array(self.shape, dtype=_np.int32)
        newshapearr = _np.array(shape, dtype=_np.int32)
        err = admm_shape_reshape(oldshapearr.size, numpy_int_buffer(oldshapearr), 
                newshapearr.size, numpy_int_buffer(newshapearr))
        if err != 0: raise ADMMError(err)
        return Expr(tuple(shape), OT_RESHAPE, [tuple(shape), self])
    
    def __getitem__(self, slc):
        '''@APIDOC(py.TensorLike.__getitem__)'''
        cdef int rdim
        shapearr = _np.array(self.shape, dtype=_np.int32)
        rshapearr = _np.empty_like(shapearr)

        nranges = 1 if isinstance(slc, (_numbers.Number, slice)) else len(slc)
        shape = self.shape

        starts = _np.empty((nranges, ), dtype=_np.int32)
        stops = _np.empty((nranges, ), dtype=_np.int32)
        steps = _np.empty((nranges, ), dtype=_np.int32)

        if nranges == 1: slc = [slc]

        for i in range(nranges):
            if isinstance(slc[i], _numbers.Number):
                # For single indices, set stops to -1 to indicate single element
                starts[i], stops[i], steps[i] =   slc[i], -1, 1
            else:
                triple = slc[i].indices(self.shape[i])
                starts[i], stops[i], steps[i] = triple
            if -starts[i] > shape[i] or starts[i] >= shape[i]:
                raise IndexError("index is out of range")
            if starts[i] < 0: starts[i] = shape[i] + starts[i]

        err = admm_shape_slice(shapearr.size, numpy_int_buffer(shapearr), starts.size,
                numpy_int_buffer(starts), numpy_int_buffer(stops), numpy_int_buffer(steps),
                &rdim, numpy_int_buffer(rshapearr))

        slices = [(starts[i], stops[i], steps[i]) for i in range(starts.size)]

        if err != 0: raise ADMMError(err)
        return Expr(tuple(rshapearr.tolist()[:rdim]), OT_SLICE, [slices, self])

    
    def __neg__(self):
        '''@APIDOC(py.TensorLike.__neg__)'''
        return Expr(tuple(self.shape), OT_NEG, [self])

    cdef _l_bin_opr(self, r, opr, matmul = False):
        cdef int rdim
        r = ensure_array_type(r)

        shapearr1 = _np.array(self.shape, dtype=_np.int32)
        shapearr2 = _np.array(r.shape, dtype=_np.int32)
        rshapearr = _np.empty((_builtins.max(shapearr1.size, shapearr2.size),), dtype=_np.int32)
        err = 0
        if matmul:
            err = admm_shape_mmul(shapearr1.size, numpy_int_buffer(shapearr1), 
                shapearr2.size, numpy_int_buffer(shapearr2),
                &rdim, numpy_int_buffer(rshapearr))
        else:
            err = admm_shape_add(shapearr1.size, numpy_int_buffer(shapearr1), 
                shapearr2.size, numpy_int_buffer(shapearr2),
                &rdim, numpy_int_buffer(rshapearr))

        if err != 0: raise ADMMError(err)
        return Expr(tuple(rshapearr.tolist()[:rdim]), opr, [self, r])

    cdef _r_bin_opr(self, l, opr, matmul = False):
        cdef int rdim
        l = ensure_array_type(l)

        shapearr1 = _np.array(l.shape, dtype=_np.int32)
        shapearr2 = _np.array(self.shape, dtype=_np.int32)

        rshapearr = _np.empty((_builtins.max(shapearr1.size, shapearr2.size),), dtype=_np.int32)

        err = 0
        if matmul:
            err = admm_shape_mmul(shapearr1.size, numpy_int_buffer(shapearr1), 
                shapearr2.size, numpy_int_buffer(shapearr2),
                &rdim, numpy_int_buffer(rshapearr))
        else:
            err = admm_shape_add(shapearr1.size, numpy_int_buffer(shapearr1), 
                shapearr2.size, numpy_int_buffer(shapearr2),
                &rdim, numpy_int_buffer(rshapearr))

        if err != 0: raise ADMMError(err)
        return Expr(tuple(rshapearr.tolist()[:rdim]), opr, [l, self])
    
    def __add__(self, r):
        '''@APIDOC(py.TensorLike.__add__)'''
        return self._l_bin_opr(r, OT_ADD)

    def __radd__(self, l):
        '''@APIDOC(py.TensorLike.__radd__)'''
        return self._r_bin_opr(l, OT_ADD)
    
    def __sub__(self, r):
        '''@APIDOC(py.TensorLike.__sub__)'''
        return self._l_bin_opr(r, OT_SUB)

    def __rsub__(self, l):
        '''@APIDOC(py.TensorLike.__rsub__)'''
        return self._r_bin_opr(l, OT_SUB)
    
    def __mul__(self, r):
        '''@APIDOC(py.TensorLike.__mul__)'''
        return self._l_bin_opr(r, OT_MUL)

    def __rmul__(self, l):
        '''@APIDOC(py.TensorLike.__rmul__)'''
        return self._r_bin_opr(l, OT_MUL)
    
    def __truediv__(self, r):
        '''@APIDOC(py.TensorLike.__truediv__)'''
        return self._l_bin_opr(r, OT_DIV)

    def __rtruediv__(self, l):
        '''@APIDOC(py.TensorLike.__truediv__)'''
        return self._r_bin_opr(l, OT_DIV)
    
    def __matmul__(self, r):
        '''@APIDOC(py.TensorLike.__matmul__)'''
        return self._l_bin_opr(r, OT_MML, True)

    def __rmatmul__(self, l):
        '''@APIDOC(py.TensorLike.__rmatmul__)'''
        return self._r_bin_opr(l, OT_MML, True)
    
    def __pow__(self, r):
        '''@APIDOC(py.TensorLike.__pow__)'''
        return self._l_bin_opr(r, OT_POW)

    def __rpow__(self, l):
        '''@APIDOC(py.TensorLike.__rpow__)'''
        return self._r_bin_opr(l, OT_POW)

    def __ge__(self, r):
        '''@APIDOC(py.TensorLike.__ge__)'''
        return Constr(self, '>', r)

    def __le__(self, r):
        '''@APIDOC(py.TensorLike.__le__)'''
        return Constr(self, '<', r)

    def __eq__(self, r):
        '''@APIDOC(py.TensorLike.__eq__)'''
        return Constr(self, '=', r)

    def __lshift__(self, zero):
        '''@APIDOC(py.TensorLike.__lshift__)'''
        if zero != 0:
            raise ValueError("rhs of an NSD constraint accepts only zero")
        return Constr(self, 'N', 0)

    def __rshift__(self, zero):
        '''@APIDOC(py.TensorLike.__rshift__)'''
        if zero != 0:
            raise ValueError("rhs of an PSD constraint accepts only zero")
        return Constr(self, 'P', 0)

    def __rlshift__(self, zero):
        '''@APIDOC(py.TensorLike.__rlshift__)'''
        if zero != 0:
            raise ValueError("rhs of an PSD constraint accepts only zero")
        return Constr(self, 'P', 0)

    def __rrshift__(self, zero):
        '''@APIDOC(py.TensorLike.__rrshift__)'''
        if zero != 0:
            raise ValueError("rhs of an NSD constraint accepts only zero")
        return Constr(self, 'N', 0)

    @constant
    def __array_priority__(self):
        return 100000000.0