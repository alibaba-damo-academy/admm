
cdef wrapPyObject(Model model, int oid, int otype):
    cdef int nslices
    if otype == AT_RANGE:
        err = admm_get_slice(model.mdl_, oid, &nslices, NULL, NULL, NULL)
        if err != 0: raise ADMMError(err)

        starts = _np.empty((nslices,), dtype=_np.int32)
        stops = _np.empty((nslices,), dtype=_np.int32)
        steps = _np.empty((nslices,), dtype=_np.int32)

        err = admm_get_slice(model.mdl_, oid, &nslices, 
            numpy_int_buffer(starts), numpy_int_buffer(stops), numpy_int_buffer(steps))
        if err != 0: raise ADMMError(err)

        slicelist = []
        for i in range(nslices):
            slicelist.append((starts[i], stops[i], steps[i]))
        return slicelist

    elif otype == AT_SPARSE:
        o = Constant(0)
        (<Constant>o).saved(model, 0, oid)
        return o
    elif otype == AT_DENSE:
        o = Constant(0)
        (<Constant>o).saved(model, 1, oid)
        return o
    elif otype == AT_PARAMETER:
        o = Param("somewhat")
        (<Param>o).model_ = model
        (<Param>o).id_ = oid
        return o
    elif otype == AT_VARIABLE:
        o = Var()
        (<Var>o).model_ = model
        (<Var>o).id_ = oid
        return o
    elif otype == AT_EXPR:
        return Expr.new(model, oid)
    else:
        raise ADMMError(10001, "UNKNOWN_ARG_TYPE")


cdef class Expr(TensorLike):
    '''@APIDOC(py.Expr)'''
    cdef Model model_
    cdef int id_
    cdef tuple shape_
    cdef int opr_
    cdef int nargs_
    cdef list operands_

    @staticmethod
    cdef new(Model model, int eid):
        expr = Expr(None, OT_NOP, [])
        expr.model_ = model
        expr.id_ = eid
        return expr
    
    def __init__(self, shape, operator, operands):
        self.id_ = -1
        self.shape_ = shape
        self.opr_ = operator
        self.nargs_ = len(operands)
        self.operands_ = list(operands)

    @constant
    def index(self):
        '''@APIDOC(py.TensorLike.index)'''
        return self.id_

    @constant
    def type(self):
        '''@APIDOC(py.TensorLike.type)'''
        return AT_EXPR

    @constant
    def operator(self):
        '''@APIDOC(py.Expr.operator)'''
        cdef int oprt
        cdef int ndim
        cdef int nargs

        if self.id_ < 0:
            oprt = self.opr_
        else:
            err = admm_get_expr(self.model_.mdl_, self.id_, &oprt, &ndim, NULL, &nargs, NULL, NULL, NULL)
            if err != 0: raise ADMMError(err)

        return oprt
        #if oprt < 100:      return OprType(oprt)
        #elif oprt < 200:    return UnaryFun(oprt)
        #elif oprt < 300:    return BinaryFun(oprt)
        #else:               return TernaryFun(oprt)

    @constant
    def nargs(self):
        '''@APIDOC(py.Expr.nargs)'''
        cdef int oprt
        cdef int ndim
        cdef int nargs
        
        if self.id_ < 0: return self.nargs_

        err = admm_get_expr(self.model_.mdl_, self.id_, &oprt, &ndim, NULL, &nargs, NULL, NULL, NULL)
        if err != 0: raise ADMMError(err)
        return nargs

    def arg(self, index):
        '''@APIDOC(py.Expr.arg)'''
        cdef int oprt
        cdef int ndim
        cdef int nargs

        if self.id_ < 0: return self.operands_[index]

        nargs = self.nargs
        args = _np.empty((nargs,), dtype=_np.int32)
        atypes = _np.empty((nargs,), dtype=_np.int32)

        err = admm_get_expr(self.model_.mdl_, self.id_, &oprt, &ndim, NULL, 
            &nargs, numpy_int_buffer(args), numpy_int_buffer(atypes), NULL)

        if err != 0: raise ADMMError(err)
        return wrapPyObject(self.model_, args[index], atypes[index])

    @constant
    def args(self):
        '''@APIDOC(py.Expr.args)'''
        return [self.arg(i) for i in range(self.nargs)]


    @constant
    def shape(self):
        '''@APIDOC(py.TensorLike.shape)'''
        cdef int oprt
        cdef int ndim
        cdef int nargs

        if self.id_ < 0: return self.shape_

        err = admm_get_expr(self.model_.mdl_, self.id_, &oprt, &ndim, NULL, &nargs, NULL, NULL, NULL)
        if err != 0: raise ADMMError(err)

        shape = _np.empty((ndim,), dtype=_np.int32)
        err = admm_get_expr(self.model_.mdl_, self.id_, &oprt, &ndim, numpy_int_buffer(shape), &nargs, NULL, NULL, NULL)
        if err != 0: raise ADMMError(err)
        return tuple(shape.tolist())

    def __repr__(self):
        li = []
        for i in range(self.nargs):
            li.append(type(self.arg(i)).__name__)
        fname = None
        if self.operator in FunInfo:
            fname = FunInfo[self.operator]
        if fname is None:
            if self.operator < len(OPNAMES):
                fname = OPNAMES[self.operator]
            else:
                for udft, udfid in udfs.items():
                    if self.operator == udfid:
                        fname = udft.__qualname__
        return "Expr({}, {}({}))".format(self.shape, fname, str(li)[1:-1])

    def __str__(self):
        li = []
        for i in range(self.nargs):
            li.append(type(self.arg(i)).__name__)
        fname = None
        if self.operator in FunInfo:
            fname = FunInfo[self.operator]
        if fname is None:
            if self.operator < len(OPNAMES):
                fname = OPNAMES[self.operator]
            else:
                for udft, udfid in udfs.items():
                    if self.operator == udfid:
                        fname = udft.__qualname__
        return "Expr({}, {}({}))".format(self.shape, fname, str(li)[1:-1])
