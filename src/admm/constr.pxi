

cdef class Constr:
    '''@APIDOC(py.Constr)'''
    cdef Model model_
    cdef int id_
    cdef object lhs_
    cdef str sense_
    cdef object rhs_

    @staticmethod
    cdef new(Model model, int cid):
        c = Constr(None, None, None)
        c.model_ = model
        c.id_ = cid
        return c

    def __init__(self, lhs, sense, rhs):
        self.id_ = -1
        self.lhs_ = lhs
        self.sense_ = sense
        self.rhs_ = rhs

    @constant
    def index(self):
        '''@APIDOC(py.Constr.index)'''
        return self.id_

    @constant
    def lhs(self):
        '''@APIDOC(py.Constr.lhs)'''
        cdef int le
        cdef int lt
        cdef int sn
        cdef int re
        cdef int rt
        if self.id_ < 0: return self.lhs_
        err = admm_get_constr(self.model_.mdl_, self.id_, &le, &lt, &sn, &re, &rt);
        if err != 0: raise ADMMError(err)
        return wrapPyObject(self.model_, le, lt)

    @constant
    def sense(self):
        '''@APIDOC(py.Constr.sense)'''
        cdef int le
        cdef int lt
        cdef int sn
        cdef int re
        cdef int rt
        if self.id_ < 0: return self.sense_
        err = admm_get_constr(self.model_.mdl_, self.id_, &le, &lt, &sn, &re, &rt);
        if err != 0: raise ADMMError(err)
        return ('<', '=', ">", 'N', 'P')[sn]

    @constant
    def rhs(self):
        '''@APIDOC(py.Constr.rhs)'''
        cdef int le
        cdef int lt
        cdef int sn
        cdef int re
        cdef int rt
        if self.id_ < 0: return self.rhs_
        err = admm_get_constr(self.model_.mdl_, self.id_, &le, &lt, &sn, &re, &rt);
        if err != 0: raise ADMMError(err)
        return wrapPyObject(self.model_, re, rt)


    def __repr__(self):
        if self.sense == 'N':
            return "Constr(NSD {})".format(self.lhs)
        elif self.sense == 'P':
            return "Constr(PSD {})".format(self.lhs)
        return "Constr({} {} {})".format(self.lhs, self.sense, self.rhs)

    def __str__(self):
        if self.sense == 'N':
            return "Constr(NSD {})".format(self.lhs)
        elif self.sense == 'P':
            return "Constr(PSD {})".format(self.lhs)
        return "Constr({} {} {})".format(self.lhs, self.sense, self.rhs)