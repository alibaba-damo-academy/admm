MINIMIZE = 1
MAXIMIZE = -1

cdef declare_rw(f):
    def fset(self, value):
        raise self.__setattr__(f.__name__, value)
    def fget(self):
        return self.__getattr__(f.__name__)
    return property(fget, fset, None, f.__doc__)

cdef declare_ro(f):
    def fset(self, value):
        raise TypeError('{} is readonly'.format(f.__name__))
    def fget(self):
        return self.__getattr__(f.__name__)
    return property(fget, fset, None, f.__doc__)

cdef class Model:
    '''@APIDOC(py.Model)'''
    cdef model_t mdl_
    cdef int needfree_
    cdef list vars_
    cdef list constrs_
    cdef list udfrefs_

    def __init__(self, obj = None, file = None, logfile=None):
        '''@APIDOC(py.Model.__init__)'''
        cdef char* name = NULL
        cdef char* filename = NULL
        cdef bytes logfilename = None

        rawname = None
        rawfilename = None
        self.needfree_ = 0
        self.vars_ = []
        self.constrs_ = []
        self.udfrefs_ = []


        if obj is not None and isinstance(obj, Model):
            self.mdl_ = admm_copy((<Model>obj).mdl_)
            self.needfree_ = 1
            self.populate()
        else:
            if logfile is not None:
                if not isinstance(logfile, str):
                    raise TypeError("expect a string for argument logfile")
                if logfile.strip() != "":
                    logfilename = bytes(logfile.strip(), "utf8")
        
        if self.needfree_ == 0:
            if obj is not None:
                if isinstance(obj, str):
                    rawname = bytes(obj, "utf8")
                elif isinstance(obj, (bytes, bytearray)):
                    rawname = obj
                else: raise ADMMError(BAD_ARG)
                name = rawname
            if file is not None:
                if isinstance(file, str):
                    rawfilename = bytes(file, "utf8")
                elif isinstance(file, (bytes, bytearray)):
                    rawfilename = file
                else: raise ADMMError(BAD_ARG)
                filename = rawfilename

            if logfilename is None:
                err = admm_create(&self.mdl_, name, filename, NULL)
            else:
                err = admm_create(&self.mdl_, name, filename, logfilename)

            if err != 0: raise ADMMError(err)

            if file is not None:
                self.populate()

            self.needfree_ = 1

    cdef populate(self):
        if len(self.vars_) > 0 or len(self.constrs_) > 0:
            raise RuntimeError("Non empty model")

        for i in range(self.numvars):
            v = Var()
            v.model_ = self
            v.id_ = i
            self.vars_.append(v)

        for j in range(self.numconstrs):
            c = Constr.new(self, j)
            self.constrs_.append(c)
        

    def getParam(self, str name):
        '''@APIDOC(py.Model.getParam)'''
        cdef int pid

        if name is None: raise ADMMError(BAD_ARG)

        rawname = bytes(name, "utf8")
        err = admm_get_param_by_name(self.mdl_, rawname, &pid)

        if err != 0: raise ADMMError(err)
        r = Param(name)
        r.model_ = self
        r.id_ = pid
        return r


    cdef addConstant(self, Constant c):
        cdef int cid
        if c.index >= 0: return c
        if c.isDense():
            data = c.data
            shape = _np.array(data.shape, dtype=_np.int32)
            err = admm_add_dense(self.mdl_, shape.size, 
                numpy_int_buffer(shape), numpy_double_buffer(data), &cid)
            if err != 0: raise ADMMError(err)
            c.saved(self, 1, cid)
        else:
            shape = _np.array(c.shape, dtype=_np.int32)
            data = c.data
            err = admm_add_sparse(self.mdl_, shape.size, numpy_int_buffer(shape), data[0].size,
                <size_t*>numpy_ulong_buffer(data[0]), numpy_double_buffer(data[1]), &cid)
            if err != 0: raise ADMMError(err)
            c.saved(self, 0, cid)
        return c


    cdef addParam(self, Param param):
        cdef int pid

        if param.index < 0:
            shape = _np.array(param.shape, dtype=_np.int32)
            name = bytes(param.name, "utf8")
            err = admm_add_param(self.mdl_, shape.size, numpy_int_buffer(shape), name, param.attr, &pid);
            if err != 0: raise ADMMError(err)
            param.id_ = pid
            param.model_ = self

        return param

    cdef addVar(self, Var var):
        cdef int vid
        cdef char* name
        namebytes = None

        if var.index < 0:
            shape = _np.array(var.shape, dtype=_np.int32)
            if var.name_ is None:
                name = NULL
            else:
                namebytes = bytes(var.name_, "utf8")
                name = namebytes
            err = admm_add_var(self.mdl_, shape.size, numpy_int_buffer(shape), name, var.attr, &vid)
            if err != 0: raise ADMMError(err)
            var.id_ = vid
            var.model_ = self
            self.vars_.append(var)
        return var

    cdef addLeaf(self, leaf):
        if isinstance(leaf, (_np.ndarray, _sp.spmatrix, tuple, list)):
            return self.addConstant(Constant(leaf))
        elif isinstance(leaf, Constant):
            return self.addConstant(leaf)
        elif isinstance(leaf, Param):
            return self.addParam(leaf)
        elif isinstance(leaf, Var):
            return self.addVar(leaf)
        else:
            return self.addConstant(Constant(leaf))

    cdef addNopExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[0])
        err = admm_add_expr_nop(self.mdl_, operand.index, operand.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addNegExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[0])
        err = admm_add_expr_neg(self.mdl_, operand.index, operand.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addTransExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[0])
        err = admm_add_expr_trans(self.mdl_, operand.index, operand.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addSliceExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[1])

        ranges = expr.operands_[0]
        starts = _np.empty((len(ranges),), dtype=_np.int32)
        stops = _np.empty((len(ranges),), dtype=_np.int32)
        steps = _np.empty((len(ranges),), dtype=_np.int32)

        for i in range(len(ranges)):
            starts[i] = ranges[i][0]
            stops[i] = ranges[i][1]
            steps[i] = ranges[i][2]

        err = admm_add_expr_slice(self.mdl_, operand.index, operand.type,
            len(expr.operands_[0]), numpy_int_buffer(starts), numpy_int_buffer(stops), 
            numpy_int_buffer(steps), &expr.id_)

        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addReshapeExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[1])

        shape = expr.operands_[0]
        shapearr = _np.array(shape, dtype=_np.int32)
        err = admm_add_expr_reshape(self.mdl_, operand.index, operand.type,
                shapearr.size, numpy_int_buffer(shapearr), &expr.id_)

        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addPlusExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_add(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addSubExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_sub(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addMulExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_mul(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addDivExpr(self, Expr expr):
        if expr.id_ >= 0: return expr
        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_div(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addMMulExpr(self, Expr expr):
        if expr.id_ >= 0: return expr

        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_matmul(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr
    
    cdef addPowExpr(self, Expr expr):
        if expr.id_ >= 0: return expr

        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_expr_pow(self.mdl_, le.index, le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addUnaryFun(self, fun, Expr expr):
        if expr.id_ >= 0: return expr
        operand = self.addExprOrConst(expr.operands_[0])

        err = admm_add_unary_fun(self.mdl_, fun, 
            operand.index, operand.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addBinaryFun(self, fun, Expr expr):
        if expr.id_ >= 0: return expr
        le = self.addExprOrConst(expr.operands_[0])
        re = self.addExprOrConst(expr.operands_[1])

        err = admm_add_binary_fun(self.mdl_, fun, le.index, 
            le.type, re.index, re.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addTernaryFun(self, fun, Expr expr):
        if expr.id_ >= 0: return expr
        o1 = self.addExprOrConst(expr.operands_[0])
        o2 = self.addExprOrConst(expr.operands_[1])
        o3 = self.addExprOrConst(expr.operands_[2])

        err = admm_add_ternary_fun(self.mdl_, fun, o1.index, 
            o1.type, o2.index, o2.type, o3.index, o3.type, &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addVariadicFun(self, fun, Expr expr):
        operands = [self.addExprOrConst(o) for o in expr.operands_]
        args = _np.array([o.index for o in operands], dtype=_np.int32)
        types = _np.array([o.type for o in operands], dtype=_np.int32)

        err = admm_add_variadic_fun(self.mdl_, fun, len(operands), numpy_int_buffer(args), numpy_int_buffer(types), &expr.id_)
        if err != 0: raise ADMMError(err)
        expr.model_ = self
        return expr

    cdef addUdfExpr(self, e):
        cdef int exprid
        udfid = getUdfId(e)
        exprs = e.arguments()
        for expr in exprs:
            if isConstExp(expr):
                raise TypeError("{}.arguments is expected to return a list of non-const expression".format(type(e).__name__))
        if len(exprs) == 0:
            raise ValueError("{}.arguments returned an empty list".format(type(e).__name__))

        exprs = [self.addExprOrConst(expr) for expr in exprs]
        indices = _np.array([expr.index for expr in exprs], dtype=_np.int32)
        etypes = _np.array([expr.type for expr in exprs], dtype=_np.int32)

        err = admm_add_expr_udf(self.mdl_, udfid, indices.size, numpy_int_buffer(indices), numpy_int_buffer(etypes), <void*>e, &exprid)
        # Increase refcount for e
        self.udfrefs_.append(e)
        if err != 0: raise ADMMError(err)
        (<UDFBase>e).index_ = exprid
        return e

    cdef addExprOrConst(self, object expr):
        cdef Expr e

        if isinstance(expr, Expr):
            e = <Expr>expr
            
            if e.opr_ == OT_NOP:
                return self.addNopExpr(e)
            elif e.opr_ == OT_NEG:
                return self.addNegExpr(e)
            elif e.opr_ == OT_T:
                return self.addTransExpr(e)
            elif e.opr_ == OT_SLICE:
                return self.addSliceExpr(e)
            elif e.opr_ == OT_RESHAPE:
                return self.addReshapeExpr(e)
            elif e.opr_ == OT_ADD:
                return self.addPlusExpr(e)
            elif e.opr_ == OT_SUB:
                return self.addSubExpr(e)
            elif e.opr_ == OT_MUL:
                return self.addMulExpr(e)
            elif e.opr_ == OT_DIV:
                return self.addDivExpr(e)
            elif e.opr_ == OT_MML:
                return self.addMMulExpr(e)
            elif e.opr_ == OT_POW:
                return self.addPowExpr(e)
            elif e.opr_ < 200:
                return self.addUnaryFun(e.opr_, e)
            elif e.opr_ < 300:
                return self.addBinaryFun(e.opr_, e)
            elif e.opr_ < 400:
                return self.addTernaryFun(e.opr_, e)
            elif e.opr_ // 100 == 9:
                # variadic function
                return self.addVariadicFun(e.opr_, e)
            else:
                raise ADMMError(BAD_ARG)
        elif isinstance(expr, UDFBase):
            return self.addUdfExpr(expr)
        else:
            return self.addLeaf(expr)

    cdef addExpr(self, expr):
        if not isinstance(expr, (Expr, UDFBase)):
            raise ADMMError(BAD_ARG)
        return self.addExprOrConst(expr)

    def addConstr(self, lexpr, sense=None, rexpr=None):
        '''@APIDOC(py.Model.addConstr)'''
        cdef int constr
        cdef Constr c

        isConstr = isinstance(lexpr, Constr)
        if isConstr:
            c = <Constr>lexpr
            if c.id_ >= 0: return lexpr
            lexpr = c.lhs
            sense = c.sense
            rexpr = c.rhs

        le = self.addExprOrConst(lexpr)
        re = self.addExprOrConst(rexpr)

        if sense not in ('<', '=', '>', 'N', 'P'): raise ADMMError(BAD_ARG)
        s = ('<', '=', '>', 'N', 'P').index(sense)
        if s < 0: raise ADMMError(BAD_ARG)

        err = admm_add_constr(self.mdl_, le.index, le.type, s, re.index, re.type, &constr)
        if err != 0: raise ADMMError(err)

        if isConstr:
            c.model_ = self
            c.id_ = constr
        else:
            c = Constr.new(self, constr)

        self.constrs_.append(c)
        return c

    def removeConstr(self, constrs):
        '''@APIDOC(py.Model.removeConstr)'''
        if isinstance(constrs, Constr):
            constrs = [constrs]
        li = _np.array([constr.index for constr in constrs], dtype=_np.int32)
        err = admm_remove_constr_list(self.mdl_, numpy_int_buffer(li), li.size)
        if err != 0: raise ADMMError(err)
        indices = sorted(li)
        i = 0
        p = 0
        offset = 0
        while i + offset < len(self.constrs_):
            if p < len(indices) and i == indices[p]:
                (<Constr>self.constrs_[i]).id_ = -1 - (<Constr>self.constrs_[i]).id_
                p += 1
                offset += 1
                while p < len(indices) and i == indices[p]: p += 1
            self.constrs_[i] = self.constrs_[i + offset]
            (<Constr>self.constrs_[i]).id_ -= offset
            i += 1
        del self.constrs_[len(self.constrs_) - offset:]

    def getVar(self, index):
        '''@APIDOC(py.Model.getVar)'''
        if index < 0 or index >= self.numvars:
            raise KeyError("index out of range <{}>".format(index))
        return self.vars_[index]

    def getVarByName(self, name):
        '''@APIDOC(py.Model.getVarByName)'''
        cdef int var
        rawname = bytes(name, "utf8")
        err = admm_get_var_by_name(self.mdl_, rawname, &var)
        if err != 0: raise ADMMError(err)
        if var >= 0: return self.getVar(var)
        return None

    def getConstr(self, index):
        '''@APIDOC(py.Model.getConstr)'''
        if index < 0 or index >= self.numconstrs:
            raise KeyError("index out of range <{}>".format(index))
        return self.constrs_[index]

    cdef containsVariable(self, expr):
        if isinstance(expr, (Var, UDFBase)):
            return True
        elif isinstance(expr, (_numbers.Number, _np.ndarray, _sp.spmatrix)):
            return False
        elif isinstance(expr, Expr):
            args = expr.args
            for arg in args:
                if self.containsVariable(arg):
                    return True
            return False
        else:
            return False

    def setObjective(self, expr, modelsense = 0):
        '''@APIDOC(py.Model.setObjective)'''
        if expr is None:
            err = admm_set_objective(self.mdl_, -1)
            if err != 0: raise ADMMError(err)
            return

        try:
            expr = ensure_array_type(expr)
        except:
            pass
        if not hasattr(expr, "shape"):
            raise TypeError("objective expects an expression or constant")
        if _np.prod(expr.shape) != 1:
            raise ValueError("objective expects a scalar")


        if not isinstance(expr, Expr) or not self.containsVariable(expr):
            expr = Expr(expr.shape, OT_NOP, [expr])

        if modelsense in [-1, 1]: self.modelsense = modelsense

        e = self.addExpr(expr)
        err = admm_set_objective(self.mdl_, e.index)
        if err != 0: raise ADMMError(err)

    def getObjective(self):
        '''@APIDOC(py.Model.getObjective)'''
        cdef int expr
        err = admm_get_objective(self.mdl_, &expr)
        if err != 0: raise ADMMError(err)
        return Expr.new(self, expr)

    def __getattr__(self, name):
        cdef int n
        cdef double f
        rawname = bytes(name.lower(), "utf8")
        if name.lower() == "minsense":
            return admm_is_minsense(self.mdl_) != 0
        elif name.lower() == "modelsense":
            return admm_model_sense(self.mdl_)
        elif name.lower() == "name":
            modelname = admm_get_model_name(self.mdl_)
            return str(modelname, "utf8") if modelname else None
        elif name.lower() in ["numvars", "numconstrs", "numiters", "status"]:
            err = admm_get_int_attr(self.mdl_, rawname, &n)
            if err != 0: raise ADMMError(err)
            return n
        elif name.lower() in ["primalobjval", "objval", "dualobjval", "solvertime", "primalgap", "dualgap"]:
            err = admm_get_double_attr(self.mdl_, rawname, &f)
            if err != 0: raise ADMMError(err)
            return f
        elif name.lower() == "statusstring":
            rawname = b'status'
            err = admm_get_int_attr(self.mdl_, rawname, &n)
            if err != 0: raise ADMMError(err)
            return [
                'SOLVE_UNKNOWN',
                'SOLVE_OPT_SUCCESS',
                'SOLVE_INFEASIBLE',
                'SOLVE_UNBOUNDED',
                'SOLVE_OVER_MAX_ITER',
                'SOLVE_OVER_MAX_TIME',
                'SOLVE_NAN_FOUND',
                'SOLVE_PRE_FAILURE',
                'SOLVE_EXCEPT_ERROR',
                'SOLVE_GET_SOL_FAILURE',
                'SOLVE_ERROR'
            ][n]
        else:
            raise AttributeError("Model has no readable attr <{}>".format(name))


    def __setattr__(self, name, value):
        if name.lower() == "minsense":
            admm_set_minsense(self.mdl_, 1 if value else 0)
        elif name.lower() == "modelsense":
            admm_set_minsense(self.mdl_, 1 if value == 1 else 0)
        else:
            raise AttributeError("Model has no writable attr <{}>".format(name))

    @declare_ro
    def NumVars(self):
        '''Number of variables in the model (read-only).'''
        pass

    @declare_ro
    def NumConstrs(self):
        '''@APIDOC(py.Model.NumConstrs)'''
        pass

    @declare_rw
    def MinSense(self):
        '''@APIDOC(py.Model.MinSense)'''
        pass

    @declare_rw
    def ModelSense(self):
        '''@APIDOC(py.Model.ModelSense)'''
        pass

    @declare_ro
    def Name(self):
        '''@APIDOC(py.Model.Name)'''
        pass

    @declare_ro
    def NumIters(self):
        '''@APIDOC(py.Model.NumIters)'''
        pass

    @declare_ro
    def Status(self):
        '''@APIDOC(py.Model.Status)'''
        pass

    @declare_ro
    def StatusString(self):
        '''@APIDOC(py.Model.StatusString)'''
        pass

    @declare_ro
    def PrimalObjVal(self):
        '''@APIDOC(py.Model.PrimalObjVal)'''
        pass

    @declare_ro
    def ObjVal(self):
        '''@APIDOC(py.Model.ObjVal)'''
        pass

    @declare_ro
    def DualObjVal(self):
        '''@APIDOC(py.Model.DualObjVal)'''
        pass

    @declare_ro
    def SolverTime(self):
        '''@APIDOC(py.Model.SolverTime)'''
        pass

    @declare_ro
    def PrimalGap(self):
        '''@APIDOC(py.Model.PrimalGap)'''
        pass

    @declare_ro
    def DualGap(self):
        '''@APIDOC(py.Model.DualGap)'''
        pass



    def setOption(self, name, value):
        '''@APIDOC(py.Model.setOption)'''
        name = name.lower()
        if name in IntOptNames:
            nval = int(value)
            err = admm_set_int_opt(self.mdl_, IntOptNames[name][0], nval)
            if err != 0: raise ADMMError(err)
        elif name.lower() in DblOptNames:
            fval = float(value)
            err = admm_set_double_opt(self.mdl_, DblOptNames[name][0], fval)
            if err != 0: raise ADMMError(err)
        else:
            raise ADMMError(OPT_NOT_FOUND)

    def getOption(self, name):
        '''@APIDOC(py.Model.getOption)'''
        cdef int nval
        cdef double fval
        name = name.lower()
        if name in IntOptNames:
            err = admm_get_int_opt(self.mdl_, IntOptNames[name][0], &nval)
            if err != 0: raise ADMMError(err)
            return nval
        elif name in DblOptNames:
            err = admm_get_double_opt(self.mdl_, DblOptNames[name][0], &fval)
            if err != 0: raise ADMMError(err)
            return fval
        else:
            raise ADMMError(OPT_NOT_FOUND)

    def getOptionInfo(self, name):
        '''@APIDOC(py.Model.getOptionInfo)'''
        if name is None: raise ADMMError(BAD_ARG)
        name = name.lower()
        if name in IntOptNames:
            return (str(IntOptNames[name][0], "utf8"), ) + IntOptNames[name][1:]
        elif name in DblOptNames:
            return (str(DblOptNames[name][0], "utf8"), ) + DblOptNames[name][1:]
        else:
            raise ADMMError(OPT_NOT_FOUND)

    def optimize(self, paramdict = None, tuner = None):
        '''@APIDOC(py.Model.optimize)'''
        cdef int pid
        cdef int rdim
        cdef tuning_callback_t cb
        cdef void* pytuner
        cdef problem_ctx_t ctx
        cdef int dense
        cdef char* errmsg

        if paramdict is None:
            paramdict = {}

        # Check all expressions
        contained_exprs = []

        try:
            contained_exprs.append(self.getObjective())
        except:
            pass

        # BFS
        for i in range(self.NumConstrs):
            c = self.getConstr(i)
            surface = len(contained_exprs)
            contained_exprs.append(c.lhs)
            contained_exprs.append(c.rhs)

            while surface < len(contained_exprs):
                cur = contained_exprs[surface]
                surface += 1

                if isinstance(cur, Expr):
                    contained_exprs += cur.args

        err = admm_create_problem_ctx(&ctx)
        if err != 0: raise ADMMError(err)

        for name in paramdict:
            rawname = bytes(name, "utf8")

            err = admm_get_param_by_name(self.mdl_, rawname, &pid)
            if err != 0:
                admm_destroy_problem_ctx(ctx)
                raise ADMMError(err)

            err = admm_get_param(self.mdl_, pid, &rdim, NULL, NULL, NULL)
            if err != 0:
                admm_destroy_problem_ctx(ctx)
                raise ADMMError(err)

            shapearr = _np.empty((rdim, ), dtype=_np.int32)
            err = admm_get_param(self.mdl_, pid, &rdim, numpy_int_buffer(shapearr), NULL, NULL)
            if err != 0:
                admm_destroy_problem_ctx(ctx)
                raise ADMMError(err)

            shape = tuple(shapearr.tolist())
            value = paramdict[name]

            if not isinstance(value, Constant):
                value = Constant(value)

            if shape != value.shape:
                admm_destroy_problem_ctx(ctx)
                raise ADMMError(SHAPE_MISMATCHED)
            
            if value.isDense():
                data = value.data
                err = admm_add_dense_binding(ctx, rawname, data.size, numpy_double_buffer(data))
            else:
                ind, val = value.data
                err = admm_add_sparse_binding(ctx, rawname, ind.size, 
                    <size_t*>numpy_ulong_buffer(ind), numpy_double_buffer(val))

            if err != 0:
                admm_destroy_problem_ctx(ctx)
                raise ADMMError(err)

        if tuner is not None:
            cb = <tuning_callback_t>ctuner
            pytuner = <void*>tuner
        else:
            cb = NULL
            pytuner = NULL

        # add start value for variables
        for var in self.vars_:
            if var.start is not None:
                start = var.start.asDense()
                npshape = _np.array(start.shape, dtype=_np.int32)
                err = admm_add_dense(self.mdl_, start.ndim, numpy_int_buffer(npshape), numpy_double_buffer(start.data), &dense)
                if err != 0: raise ADMMError(err)
                err = admm_set_variable_start(self.mdl_, var.index, dense)
                if err != 0: raise ADMMError(err)

        if tuner is not None:
            tstate = pysavethread()
        try:
            err = admm_optimize(self.mdl_, ctx, cb, pytuner, &errmsg)
        finally:
            if tuner is not None: pyrestorethread(tstate)

        admm_destroy_problem_ctx(ctx)
        if err != 0:
            e = ADMMError(err)
            if e.message == 'OPTIMIZER_ERROR':
                raise ADMMError(err, str(errmsg, "utf-8"))
            raise e

    def write(self, fname):
        '''@APIDOC(py.Model.write)'''
        err = admm_model_write(self.mdl_, bytes(fname, "utf8"))
        if err != 0: raise ADMMError(err)

    def dispose(self):
        '''@APIDOC(py.Model.dispose)'''
        if self.needfree_ == 1:
            self.needfree_ = 0
            self.udfrefs_ = []
            admm_destroy(self.mdl_)

    def __del__(self):
        self.dispose()

    def __dealloc__(self):
        self.dispose()