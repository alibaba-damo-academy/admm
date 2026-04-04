cdef int nhypers
cdef int dhypers
cdef list inthyperinfo = []
cdef list dblhyperinfo = []
cdef list allhypernames = []
cdef dict inthypermap = {}
cdef dict dblhypermap = {}
cdef dict hyperprops = {}

cdef loadHypers():
    cdef const char* name
    cdef int writable
    cdef int nmin
    cdef int nmax
    cdef int ndef
    cdef double dmin
    cdef double dmax
    cdef double ddef
    cdef const char* cndoc
    cdef const char* endoc

    global nhypers
    global dhypers
    global inthyperinfo
    global dblhyperinfo
    global allhypernames
    global inthypermap
    global dblhypermap
    global hyperprops

    nhypers = admm_num_int_hypers()
    dhypers = admm_num_double_hypers()
    inthyperinfo = []
    dblhyperinfo = []
    allhypernames = []
    inthypermap = {}
    dblhypermap = {}
    hyperprops = {}

    exist = set()

    for i in range(nhypers):
        err = admm_get_int_hyper_info(i, &name, &writable, &nmin, &nmax, &ndef, &cndoc, &endoc);
        if err != 0: raise ADMMError(err)
        inthyperinfo.append((str(name, "utf8"), writable, nmin, nmax, ndef, str(cndoc, "utf8"), str(endoc, "utf8")))

        hypername = inthyperinfo[-1][0]
        exist.add(hypername)
        allhypernames.append(hypername)
        inthypermap[hypername] = i

        hyperprops[hypername] = property(
            fset=lambda self, value, ctx=hypername: self.__setattr__(ctx, value),
            fget=lambda self, ctx=hypername: self.__getattr__(ctx),
            doc="type=int, min={}, max={}, writable={}\n{}".format(inthyperinfo[-1][2], inthyperinfo[-1][3], inthyperinfo[-1][1]==1, inthyperinfo[-1][6])
        )
        

    for i in range(dhypers):
        err = admm_get_double_hyper_info(i, &name, &writable, &dmin, &dmax, &ddef, &cndoc, &endoc);
        if err != 0: raise ADMMError(err)
        dblhyperinfo.append((str(name, "utf8"), writable, dmin, dmax, ddef, str(cndoc, "utf8"), str(endoc, "utf8")))

        hypername = dblhyperinfo[-1][0]
        exist.add(hypername)
        allhypernames.append(hypername)
        dblhypermap[hypername] = i
        hyperprops[hypername] = property(
            fset=lambda self, value, ctx=hypername: self.__setattr__(ctx, value),
            fget=lambda self, ctx=hypername: self.__getattr__(ctx),
            doc="type=float, min={}, max={}, writable={}\n{}".format(dblhyperinfo[-1][2], dblhyperinfo[-1][3], dblhyperinfo[-1][1]==1, dblhyperinfo[-1][6])
        )

loadHypers()

cdef class TuningContext:
    cdef int* nactive_
    cdef int* dactive_
    cdef int* nval_
    cdef double* dval_

    def __metacls__(self):
        loadHypers()

        class MetaClass(type):
            # this meta class is to pretty pydoc for TuningContext
            def __dir__(self):
                global allhypernames
                return allhypernames

            def __getattr__(self, name):
                global hyperprops
                return hyperprops.get(name)

        return MetaClass

    def __init__(self):
        pass

    cdef bind(self, int* nactive, int* dactive, int* nval, double* dval):
        self.nactive_ = nactive
        self.dactive_ = dactive
        self.nval_ = nval
        self.dval_ = dval

        global inthyperinfo
        global dblhyperinfo

        for i in range(len(inthyperinfo)):
            self.nactive_[i] = 0

        for i in range(len(dblhyperinfo)):
            self.dactive_[i] = 0

    def __iter__(self):
        global allhypernames
        return iter(allhypernames)

    def __len__(self):
        global allhypernames
        return len(allhypernames)

    def __getattr__(self, name):
        global inthypermap
        global dblhypermap

        nid = inthypermap.get(name.lower())
        did = dblhypermap.get(name.lower())

        if nid is not None:
            return int(self.nval_[nid])

        if did is not None:
            return float(self.dval_[did])

        raise AttributeError("property `{}` not found".format(name))

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        global inthyperinfo
        global dblhyperinfo
        global inthypermap
        global dblhypermap

        nid = inthypermap.get(name.lower())
        did = dblhypermap.get(name.lower())

        if nid is not None:
            nval = int(value)
            if nval < inthyperinfo[nid][2] or nval > inthyperinfo[nid][3]:
                raise ValueError("property value out of bound")
            if inthyperinfo[nid][1] == 0:
                raise KeyError("property is readonly")
            self.nactive_[nid] = 1
            self.nval_[nid] = value
            return

        if did is not None:
            dval = float(value)
            if dval < dblhyperinfo[did][2] or dval > dblhyperinfo[did][3]:
                raise ValueError("property value out of bound")
            if dblhyperinfo[did][1] == 0:
                raise KeyError("property is readonly")
            self.dactive_[did] = 1
            self.dval_[did] = value
            return

        raise AttributeError("property `{}` not found".format(name))

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __repr__(self):
        global allhypernames
        li = [name + " = " + str(self[name]) for name in allhypernames]
        return "TuningContext({})".format(", ".join(li))

    def __str__(self):
        return self.__repr__()


cdef void _tune(object pyfun, int* nactive, int* dactive, int* nval, double* dval):
    with (callbackLock):
        ctx = TuningContext()
        ctx.bind(nactive, dactive, nval, dval)
        try:
            pyfun(ctx)
        except Exception as e:
            _traceback.print_exc()

cdef void ctuner(object pyfun, int* nactive, int* dactive, int* nval, double* dval):
    gillock = pygillock()
    try:
        # Never inline the following function call
        # Otherwise python may raise error 'GIL released'
        _tune(pyfun, nactive, dactive, nval, dval)
    finally:
        pygilrelease(gillock)
