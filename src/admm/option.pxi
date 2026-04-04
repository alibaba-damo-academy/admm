# Option metadata

cdef dict IntOptNames = {}
cdef dict DblOptNames = {}

class OptionConstClass:
    pass

cdef loadOptMeta():
    cdef int minn
    cdef int maxn
    cdef int defn
    cdef double minf
    cdef double maxf
    cdef double deff
    cdef char* name
    cdef const char* doc

    numIntOpts = admm_num_int_opts()
    numDblOpts = admm_num_double_opts()
    for i in range(numIntOpts):
        err = admm_get_int_opt_info(i, <const char**>&name, NULL, &minn, &maxn, &defn, NULL, &doc)
        if err != 0: raise ADMMError(err)
        optname = str(name, "utf8")
        IntOptNames[optname.lower()] = (name, minn, maxn, defn)
        setattr(OptionConstClass, optname, property(
            # Use ctx to store temp variable
            fget = lambda self, ctx = optname: ctx,
            doc = "type = int, min = {}, max = {}, default = {}\n{}".format(minn, maxn, defn, str(doc, "utf8"))
        ))

    for i in range(numDblOpts):
        err = admm_get_double_opt_info(i, <const char**>&name, NULL, &minf, &maxf, &deff, NULL, &doc)
        if err != 0: raise ADMMError(err)
        optname = str(name, "utf8")
        DblOptNames[optname.lower()] = (name, minf, maxf, deff)
        setattr(OptionConstClass, optname, property(
            # Use ctx to store temp variable
            fget = lambda self, ctx = optname: ctx,
            doc = "type = double, min = {}, max = {}, default = {}\n{}".format(minf, maxf, deff, str(doc, "utf8"))
        ))

Options = OptionConstClass()

loadOptMeta()
