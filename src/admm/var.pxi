
cdef class Var(TensorLike):
    '''@APIDOC(py.Var)'''
    cdef Model model_
    cdef int id_
    cdef str name_
    cdef tuple shape_
    cdef int attr_
    cdef Constant start_

    def __init__(self, *shape, **attr):
        '''@APIDOC(py.Var.__init__)'''
        if len(shape) > 0 and isinstance(shape[0], (str, bytes, bytearray)):
            # Variable name specified
            name = shape[0]
            if isinstance(name, (bytes, bytearray)):
                name = str(name, "utf8")
            if name.startswith('$'):
                raise ValueError('user specified variable name must not start with $')
            for c in name:
                if not (c.isalnum() or c == '_'):
                    raise ValueError(f'variable name must contain only letters, digits, and underscores (got {c!r})')
            self.name_ = name
            self.shape_ = tuple(shape[1:])
        else:
            self.name_ = None
            self.shape_ = tuple(shape)

        # Unwrap single-element tuple shape: Var('X', (3, 4)) → shape_ = (3, 4)
        if len(self.shape_) == 1 and isinstance(self.shape_[0], (tuple, list)):
            self.shape_ = tuple(self.shape_[0])

        # Validate dimensions: must be non-negative integers
        _validated = []
        for _d in self.shape_:
            try:
                _di = int(_d)
            except (TypeError, ValueError):
                raise TypeError("variable dimensions must be integers, got {}".format(type(_d).__name__))
            if _di <= 0:
                raise ValueError("variable dimensions must be positive, got {}".format(_di))
            _validated.append(_di)
        self.shape_ = tuple(_validated)

        self.attr_ = 0
        self.id_ = -1

        for key in attr:
            if key not in tensorattrnames:
                raise KeyError("unknown attribute `{}`".format(key))
            if not attr[key]: continue
            if self.attr_ != 0:
                raise ValueError("accept only 1 attribute")
            self.attr_ = 1 << tensorattrnames[key]

        self.start_ = None

    @property
    def start(self):
        '''@APIDOC(py.Var.start)'''
        return self.start_

    @start.setter
    def start(self, value):
        self.start_ = Constant(value)
        if self.start_.shape != self.shape:
            self.start_ = None
            raise ValueError("mismatched shape")

    @constant
    def shape(self):
        '''@APIDOC(py.TensorLike.shape)'''
        cdef int dim

        if self.index < 0:
            return self.shape_

        err = admm_get_var(self.model_.mdl_, self.id_, &dim, NULL, NULL, NULL);
        if err != 0: raise ADMMError(err)

        shape = _np.empty((dim, ), dtype=_np.int32)
        err = admm_get_var(self.model_.mdl_, self.id_, &dim, numpy_int_buffer(shape), NULL, NULL);
        if err != 0: raise ADMMError(err)

        return tuple(shape.tolist())

    @constant
    def attr(self):
        '''@APIDOC(py.Var.attr)'''
        cdef int attr
        if self.index < 0:
            return self.attr_
        err = admm_get_var(self.model_.mdl_, self.id_, NULL, NULL, NULL, &attr)
        if err != 0: raise ADMMError(err)
        return attr

    @constant
    def X(self):
        '''@APIDOC(py.Var.X)'''
        if self.model_ is None:
            raise ADMMError("Variable is not associated with any model. "
                            "Add it to a model via setObjective/addConstr, then call optimize() first.")
        shape = self.shape
        sol = _np.empty(shape, dtype=_np.float64)
        err = admm_get_var_solution(self.model_.mdl_, self.id_, numpy_double_buffer(sol))
        if err != 0: raise ADMMError(err)
        return sol

    @constant
    def index(self):
        '''@APIDOC(py.TensorLike.index)'''
        return self.id_

    @constant
    def name(self):
        '''@APIDOC(py.Param.name)'''
        cdef char* name

        if self.index < 0:
            return self.name_

        err = admm_get_var(self.model_.mdl_, self.id_, NULL, NULL, &name, NULL);
        if err != 0: raise ADMMError(err)
        return str(name, "utf8")

    @constant
    def type(self):
        '''@APIDOC(py.TensorLike.type)'''
        return AT_VARIABLE

    def __repr__(self):
        name = self.name
        return "Var({}, {})".format("unnamed" if name is None else name, self.shape)

    def __str__(self):
        name = self.name
        return "Var({}, {})".format("unnamed" if name is None else name, self.shape)