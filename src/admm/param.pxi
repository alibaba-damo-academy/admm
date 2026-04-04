
cdef class Param(TensorLike):
    '''@APIDOC(py.Param)'''
    cdef Model model_
    cdef int id_
    cdef str name_
    cdef tuple shape_
    cdef int attr_

    def __init__(self, name, *shape, **attr):
        '''@APIDOC(py.Param.__init__)'''
        if not isinstance(name, str): raise TypeError("name expect a string")
        if name is None or len(name.strip()) == 0: raise ValueError("invalid value for name")
        if not isinstance(shape, tuple): raise TypeError("shape expect a tuple")

        for c in name:
            if not (c.isalnum() or c == '_'):
                raise ValueError(f'parameter name must contain only letters, digits, and underscores (got {c!r})')

        self.name_ = name

        self.shape_ = shape

        # Unwrap single-element tuple shape: Param('p', (3, 4)) → shape_ = (3, 4)
        if len(self.shape_) == 1 and isinstance(self.shape_[0], (tuple, list)):
            self.shape_ = tuple(self.shape_[0])

        # Validate dimensions: must be positive integers
        _validated = []
        for _d in self.shape_:
            try:
                _di = int(_d)
            except (TypeError, ValueError):
                raise TypeError("parameter dimensions must be integers, got {}".format(type(_d).__name__))
            if _di <= 0:
                raise ValueError("parameter dimensions must be positive, got {}".format(_di))
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

    @constant
    def shape(self):
        '''@APIDOC(py.TensorLike.shape)'''
        cdef int dim

        if self.index < 0:
            return self.shape_

        err = admm_get_param(self.model_.mdl_, self.id_, &dim, NULL, NULL, NULL);
        if err != 0: raise ADMMError(err)

        shape = _np.empty((dim, ), dtype=_np.int32)
        err = admm_get_param(self.model_.mdl_, self.id_, &dim, numpy_int_buffer(shape), NULL, NULL);
        if err != 0: raise ADMMError(err)

        return tuple(shape.tolist())        

    @constant
    def attr(self):
        '''@APIDOC(py.Param.attr)'''
        cdef int attr
        if self.index < 0:
            return self.attr_
        err = admm_get_param(self.model_.mdl_, self.id_, NULL, NULL, NULL, &attr)
        if err != 0: raise ADMMError(err)
        return attr

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

        err = admm_get_param(self.model_.mdl_, self.id_, NULL, NULL, &name, NULL);
        if err != 0: raise ADMMError(err)
        return str(name, "utf8")

    @constant
    def type(self):
        '''@APIDOC(py.TensorLike.type)'''
        return AT_PARAMETER
        
    def __repr__(self):
        return "Param({}, {})".format(self.name, self.shape)

    def __str__(self):
        return "Param({}, {})".format(self.name, self.shape)