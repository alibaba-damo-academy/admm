cdef isConstExp(expr):
    return not isinstance(expr, (Param, Var, UDFBase, Expr))

class OptionConstClass:
    '''@APIDOC(py.OptionConstClass)'''
{$ for nopt in meta[0] $}
    @constant
    def {$ nopt.name $}(self):
        '''type: int\nmin: {$ nopt.min $}, max: {$ nopt.max $}, default: {$ nopt.default $}\n{$ nopt.endoc $}'''
        return "{$ nopt.name $}"

{$ end $}

{$ for dopt in meta[1] $}
    @constant
    def {$ dopt.name $}(self):
        '''type: float\nmin: {$ dopt.min $}, max: {$ dopt.max $}, default: {$ dopt.default $}\n{$ dopt.endoc $}'''
        return "{$ dopt.name $}"

{$ end $}

Options = OptionConstClass()

cdef IntOptNames = {
{$ for nopt in meta[0] $}
    "{$ nopt.name.lower() $}": (b"{$ nopt.name $}", {$ nopt.min $}, {$ nopt.max $}, {$ nopt.default $}),
{$ end $}
}


cdef DblOptNames = {
{$ for dopt in meta[1] $}
    "{$ dopt.name.lower() $}": (b"{$ dopt.name $}", {$ dopt.min $}, {$ dopt.max $}, {$ dopt.default $}),
{$ end $}
}

cdef FunInfo = {
{$ for fun in meta[4] $}
    {$ fun.opr $}: "{$ fun.name $}",
{$ end $}
}

__all__ += {$ str(list(set(list(map(lambda f: f.name, meta[4]))))) $}

cdef class TuningContext:
    '''@APIDOC(py.TuningContext)'''
    cdef int* nactive_
    cdef int* dactive_
    cdef int* nval_
    cdef double* dval_


    def __metacls__(self):

        class MetaClass(type):
            # this meta class is to pretty pydoc for TuningContext
            def __dir__(self):
                {$ nnames = list(map(lambda h: h.name, meta[2])) $}
                {$ dnames = list(map(lambda h: h.name, meta[3])) $}
                return {$ str(nnames + dnames) $}

        return MetaClass

    cdef bind(self, int* nactive, int* dactive, int* nval, double* dval):
        self.nactive_ = nactive
        self.dactive_ = dactive
        self.nval_ = nval
        self.dval_ = dval

        nhypers = {$ len(meta[2]) $}
        dhypers = {$ len(meta[3]) $}

        for i in range(nhypers):
            self.nactive_[i] = 0

        for i in range(dhypers):
            self.dactive_[i] = 0

{$ for i in range(len(meta[2])) $}
    {$ nhyper = meta[2][i] $}

    @property
    def {$ nhyper.name $}(self):
        '''type: int\nmin: {$ nhyper.min $}, max: {$ nhyper.max $}\n{$ nhyper.endoc $}'''
        return self.nval_[{$ i $}]

    @{$ nhyper.name $}.setter
    def {$ nhyper.name $}(self, value):
        if not isinstance(value, _numbers.Number):
            raise ValueError('expect an numeric value')
        if value < {$ nhyper.min $} or value > {$ nhyper.max $}:
            raise ValueError('value is out of range')
        self.nactive_[{$ i $}] = 1
        self.nval_[{$ i $}] = int(value)

{$ end $}


{$ for i in range(len(meta[3])) $}
    {$ dhyper = meta[3][i] $}
    
    @property
    def {$ dhyper.name $} (self):
        '''type: float\nmin: {$ dhyper.min $}, max: {$ dhyper.max $}\n{$ dhyper.endoc $}'''
        return self.dval_[{$ i $}]

    @{$ dhyper.name $}.setter
    def {$ dhyper.name $}(self, value):
        if not isinstance(value, _numbers.Number):
            raise ValueError('expect an numeric value')
        if value < {$ dhyper.min $} or value > {$ dhyper.max $}:
            raise ValueError('value is out of range')
        self.dactive_[{$ i $}] = 1
        self.dval_[{$ i $}] = float(value)

{$ end $}


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


cdef _unary_fun_impl(fun, expr):
    cdef int rdim
    cdef Constant c
    
    expr = ensure_array_type(expr)
    shapearr = _np.array(expr.shape, dtype=_np.int32)

    err = admm_shape_unary_fun(fun, shapearr.size, 
        numpy_int_buffer(shapearr), &rdim, NULL)
    if err != 0: raise ADMMError(err)

    rshapearr = _np.zeros(rdim, dtype=_np.int32)
    err = admm_shape_unary_fun(fun, shapearr.size, 
        numpy_int_buffer(shapearr), &rdim, numpy_int_buffer(rshapearr))

    if err != 0: raise ADMMError(err)

    if isConstExp(expr):
        c = Constant(expr)
        err = admm_tensor_unary_fun(fun, c.tensor_)
        if err != 0: raise ADMMError(err)
        return c

    return Expr(tuple(rshapearr.tolist()[:rdim]), fun, [expr])

cdef _binary_fun_impl(fun, expr1, expr2):
    cdef int rdim
    cdef Constant t1
    cdef Constant t2
    
    expr1 = ensure_array_type(expr1)
    expr2 = ensure_array_type(expr2)

    shapearr1 = _np.array(expr1.shape, dtype=_np.int32)
    shapearr2 = _np.array(expr2.shape, dtype=_np.int32)

    err = admm_shape_binary_fun(fun, shapearr1.size, 
        numpy_int_buffer(shapearr1), shapearr2.size, 
        numpy_int_buffer(shapearr2), &rdim, NULL)
    if err != 0: raise ADMMError(err)

    rshapearr = _np.zeros(rdim, dtype=_np.int32)
    err = admm_shape_binary_fun(fun, shapearr1.size, 
        numpy_int_buffer(shapearr1), shapearr2.size, 
        numpy_int_buffer(shapearr2), &rdim, numpy_int_buffer(rshapearr))
    if err != 0: raise ADMMError(err)


    if isConstExp(expr1) and isConstExp(expr2):
        t1 = Constant(expr1)
        t2 = Constant(expr2)
        err = admm_tensor_binary_fun(fun, t1.tensor_, t2.tensor_)
        if err != 0: raise ADMMError(err)
        return t1

    return Expr(tuple(rshapearr.tolist()[:rdim]), fun, [expr1, expr2])

cdef _ternary_fun_impl(fun, expr1, expr2, expr3):
    cdef int rdim
    cdef Constant t1
    cdef Constant t2
    cdef Constant t3

    expr1 = ensure_array_type(expr1)
    expr2 = ensure_array_type(expr2)
    expr3 = ensure_array_type(expr3)
    
    shapearr1 = _np.array(expr1.shape, dtype=_np.int32)
    shapearr2 = _np.array(expr2.shape, dtype=_np.int32)
    shapearr3 = _np.array(expr3.shape, dtype=_np.int32)

    err = admm_shape_ternary_fun(fun, shapearr1.size, 
        numpy_int_buffer(shapearr1), shapearr2.size, 
        numpy_int_buffer(shapearr2), shapearr3.size, 
        numpy_int_buffer(shapearr3), &rdim, NULL)
    if err != 0: raise ADMMError(err)

    rshapearr = _np.zeros(rdim, dtype=_np.int32)
    err = admm_shape_ternary_fun(fun, shapearr1.size, 
        numpy_int_buffer(shapearr1), shapearr2.size, 
        numpy_int_buffer(shapearr2), shapearr3.size, 
        numpy_int_buffer(shapearr3), &rdim, numpy_int_buffer(rshapearr))
    if err != 0: raise ADMMError(err)

    if isConstExp(expr1) and isConstExp(expr2) and isConstExp(expr3):
        t1 = Constant(expr1)
        t2 = Constant(expr2)
        t3 = Constant(expr3)
        err = admm_tensor_ternary_fun(fun, t1.tensor_, t2.tensor_, t3.tensor_)
        if err != 0: raise ADMMError(err)
        return t1

    return Expr(tuple(rshapearr.tolist()[:rdim]), fun, [expr1, expr2, expr3])


{$

class FunctionGroup:

    def __init__(self, funs):
        self.funs = sorted(funs, key=lambda o: o.opr)
        self.name = funs[0].name
        self.const_args = funs[0].const_args

    def signature(self, default=True):
        argnames = ["x{}".format(i + 1) for i in range(self.funs[0].nargs)]
        if len(argnames) == 1:
            argnames[0] = "x"

        if self.funs[0].args.strip() == "":
            return ", ".join(argnames)
        max_arg_count = max(len(fun.args.split(",")) for fun in self.funs)
        argnames = [None for i in range(max_arg_count)]

        for fun in self.funs:
            args = fun.args.strip().split(",")
            for i in range(len(args)):
                arg = args[i].strip()
                if "=" in arg:
                    if default:
                        argnames[i] = arg.strip()
                    else:
                        argnames[i] = arg[:arg.index("=")].strip()
                elif argnames[i] is None:
                    argnames[i] = arg.strip()

        return ", ".join(argnames)

    def branches(self):
        if len(self.funs) == 1:
            prefix = ["unary", "binary", "ternary"][self.funs[0].nargs - 1]
            fcall = "_{}_fun_impl({}, {})".format(prefix, self.funs[0].opr, self.signature(False))
            return [(None, fcall)]

        branches = []
        defaults = []
        for f in self.funs:
            cond = []
            argnames = []
            raw_args = [a.strip() for a in f.args.split(",") if a.strip() != ""]
            for i, raw in enumerate(raw_args):
                arg = raw.strip()
                if arg.startswith("const "):
                    arg = arg[len("const "):].strip()
                name = arg
                default = None
                if ":" in arg:
                    name, default = arg.split(":", 1)
                elif "=" in arg:
                    name, default = arg.split("=", 1)
                name = name.strip()
                if i >= f.nargs and default is not None:
                    cond.append("{}=={}".format(name, default.strip()))
                argnames.append(name)

            vars = argnames[:f.nargs]
            prefix = ["unary", "binary", "ternary"][f.nargs - 1]
            fcall = "_{}_fun_impl({}, {})".format(prefix, f.opr, ", ".join(vars))
            if len(cond) == 0:
                defaults.append((f.nargs, fcall))
            else:
                branches.append((" and ".join(cond), fcall))

        if len(defaults) > 0:
            # Pick the most specific default (largest nargs)
            defaults.sort(key=lambda item: item[0], reverse=True)
            branches.append((None, defaults[0][1]))

        return branches


class FunctionList:

    def __init__(self, functions):
        groups = {}
        for f in functions:
            if f.name not in groups:
                groups[f.name] = [f]
            else:
                groups[f.name].append(f)
        self.groups = [FunctionGroup(group) for group in groups.values()]

    def __iter__(self):
        return iter(self.groups)

funGroups = FunctionList(meta[4])

$}


{$ for fun in funGroups $}
    {$ if fun.funs[0].opr // 100 != 9 $}
def {$ fun.name $}({$ fun.signature() $}):
    '''@APIDOC(py.{$ fun.name $})'''
    {$

    # Check arguments, some of them can only be constant expression

    $}
    {$ argnames = [arg.strip() for arg in fun.signature(False).split(",")] $}
    {$ for i in range(len(fun.const_args)) $}
    {$ if fun.const_args[i] == '1' $}
    if isinstance({$ argnames[i] $}, (Var, Expr)):
        raise TypeError("constant expression is expected for argument `{$ argnames[i] $}`")
    {$ end $}
    {$ end $}
    {$

    # Check arguments, some of them can only be constant expression
    
    $}
    {$ branches = fun.branches() $}
    {$ if len(branches) == 1 and branches[0][0] is None $}
    return {$ branches[0][1] $}
    {$ else $}
    {$ for cond, fcall in branches $}
    {$ if cond is None $}
    return {$ fcall $}
    {$ else $}
    if {$ cond $}: return {$ fcall $}
    {$ end $}
    {$ end $}
    raise ValueError("unexpected argument(s)")
    {$ end $}
    {$ end $}
{$ end $}