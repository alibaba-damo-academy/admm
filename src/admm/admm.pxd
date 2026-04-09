# cython: language_level=3

cdef extern from "metaclass.h":
    pass

cdef extern from "admm.h":
    ctypedef void (*tuning_callback_t)(void* userdata, int* iactive, int* dactive, int* intval, double* dblval);
    ctypedef int (*value_eval_t)(void* userdata, int nvars, int* ndims, int* shapes, double* tensors, double* result);
    ctypedef int (*argmin_t)(void* userdata, double lamb, int nvars, int* ndims, int* shapes, double* tensors);
    ctypedef int (*grad_t)(void* userdata, int nvars, int* ndims, int* shapes, double* tensors, double* grad_out);
    ctypedef void* model_t;
    ctypedef void* tensor_t;
    ctypedef void* problem_ctx_t;

    double admm_get_epsilon();
    double admm_set_epsilon(double epsilon);

    const char* admm_explain_err(int code);
    int admm_error_code(const char* msg);

    int admm_custom_udf(const char* name, value_eval_t eval, argmin_t argmin, int* udf);
    int admm_custom_udf_with_grad(const char* name, value_eval_t eval, grad_t grad, int* udf);
    int admm_get_udf(int udf, char** name, value_eval_t* eval, argmin_t* argmin);
    int admm_get_udf_grad(int udf, grad_t* grad);
    int admm_is_udf_opr(int opr);

    int admm_shape_add(int ndim1, int* shape1, int ndim2, int* shape2, int* rdim, int* rshape);
    int admm_shape_mmul(int ndim1, int* shape1, int ndim2, int* shape2, int* rdim, int* rshape);
    int admm_shape_t(int ndim, int* shape, int* rdim, int* rshape);
    int admm_shape_slice(int ndim, int* shape, int n, int* starts, int* stops, int* steps, int* rdim, int* rshape);
    int admm_shape_reshape(int ndim, int* shape, int newdim, int* newshape);

    int admm_shape_unary_fun(int fun, int ndim, int* shape, int* rdim, int* rshape);
    int admm_shape_binary_fun(int fun, int ndim1, int* shape1, int ndim2, int* shape2, int* rdim, int* rshape);
    int admm_shape_ternary_fun(int fun, int ndim1, int* shape1, int ndim2, int* shape2, int ndim3, int* shape3, int* rdim, int* rshape);
    int admm_shape_variadic_fun(int fun, int nopr, int* ndim, int** shape, int* rdim, int* rshape);

    int admm_create(model_t* model, const char* name, const char* filename, const char* logfilename);
    model_t admm_copy(model_t model);
    void admm_destroy(model_t model);
    const char* admm_get_model_name(model_t model);
    int admm_add_sparse(model_t model, int ndim, int* shape, int nzs, size_t* inds, double* vals, int* sparse);
    int admm_add_dense(model_t model, int ndim, int* shape, double* elements, int* dense);
    int admm_get_sparse(model_t model, int sparse, int* ndim, int* shape, int* nzs, size_t* inds, double* vals);
    int admm_get_dense(model_t model, int dense, int* ndim, int* shape, double* elements);
    int admm_add_param(model_t model, int ndim, int* shape, const char* name, int attr, int* parameter);
    int admm_get_param(model_t model, int param, int* ndim, int* shape, char** name, int* attr);
    int admm_get_param_by_name(model_t model, const char* name, int* param);
    int admm_add_var(model_t model, int ndim, int* shape, const char* name, int attr, int* var);
    int admm_get_var(model_t model, int var, int* ndim, int* shape, char** name, int* attr);
    int admm_get_var_by_name(model_t model, const char* name, int* var);
    int admm_set_variable_start(model_t model, int var, int dense);
    int admm_get_variable_start(model_t model, int var, int* dense);
    int admm_add_expr_trans(model_t model, int source, int type, int* expr);
    int admm_add_expr_nop(model_t model, int source, int type, int* expr);
    int admm_add_expr_neg(model_t model, int source, int type, int* expr);
    int admm_add_expr_slice(model_t model, int source, int type, int nslices, int* starts, int* stops, int* steps, int* expr);
    int admm_add_expr_reshape(model_t model, int source, int type, int ndim, int* newshape, int* expr);
    int admm_add_expr_add(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_sub(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_mul(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_div(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_pow(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_matmul(model_t model, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_expr_udf(model_t model, int udf, int nargs, int* args, int* types, void* userdata, int* expr);
    int admm_get_udf_ctx(model_t model, int expr, void** userdata);
    int admm_get_slice(model_t model, int slice, int* nslices, int* starts, int* stops, int* steps);
    int admm_get_shape(model_t, int shape, int* ndim, int* buffer);
    int admm_get_expr(model_t model, int expr, int* oprType, int* ndim, int* shape, int* nargs, int* args, int* atypes, int* feature);

    int admm_add_unary_fun(model_t model, int fun, int arg, int type, int* expr);
    int admm_add_binary_fun(model_t model, int fun, int arg1, int type1, int arg2, int type2, int* expr);
    int admm_add_ternary_fun(model_t model, int fun, int arg1, int type1, int arg2, int type2, int arg3, int type3, int* expr);
    int admm_add_variadic_fun(model_t model, int fun, int nargs, int* args, int* types, int* expr);

    void admm_print(model_t model, const char* table, int start, int len);

    # Model operations

    int admm_add_constr(model_t model, int lhs, int lhst, int sense, int rhs, int rhst, int* constr);
    int admm_remove_constr_list(model_t model, int* indices, int len);
    int admm_remove_constr_range(model_t model, int start, int len);
    int admm_get_constr(model_t model, int constr, int* lhs, int* lhst, int* sense, int* rhs, int* rhst);
    int admm_set_objective(model_t model, int expr);
    int admm_get_objective(model_t model, int* expr);
    int admm_model_sense(model_t model);
    void admm_set_minsense(model_t model, int minSense);
    int admm_is_minsense(model_t model);

    # Option operations

    int admm_set_int_opt(model_t model, const char* name, int value);
    int admm_set_double_opt(model_t model, const char* name, double value);
    int admm_get_int_opt(model_t model, const char* name, int* value);
    int admm_get_double_opt(model_t model, const char* name, double* value);
    int admm_num_int_opts();
    int admm_num_double_opts();
    int admm_get_int_opt_info(int opt, const char** name, int* writable, int* min, int* max, int* defv, const char** cndoc, const char** endoc);
    int admm_get_double_opt_info(int opt, const char** name, int* writable, double* min, double* max, double* defv, const char** cndoc, const char** endoc);
    void admm_reset_opts(model_t model);

    # Hypers

    int admm_num_int_hypers();
    int admm_num_double_hypers();
    int admm_get_int_hyper_info(int hyper, const char** name, int* writable, int* min, int* max, int* defv, const char** cndoc, const char** endoc);
    int admm_get_double_hyper_info(int hyper, const char** name, int* writable, double* min, double* max, double* defv, const char** cndoc, const char** endoc);

    # Optimize
    int admm_create_problem_ctx(problem_ctx_t* ctx);
    void admm_destroy_problem_ctx(problem_ctx_t ctx);
    int admm_add_sparse_binding(problem_ctx_t ctx, const char* pname, int nzs, size_t* inds, double* vals);
    int admm_add_dense_binding(problem_ctx_t ctx, const char* pname, int len, double* data);
    int admm_optimize(model_t model, problem_ctx_t ctx, tuning_callback_t tuner, void* userdata, char** errmsg);

    # IO
    int admm_model_write(model_t model, const char* file);
    int admm_model_read(model_t model, const char* file);

    # Attributes
    int admm_get_int_attr(model_t model, const char* name, int* attrval);
    int admm_get_double_attr(model_t model, const char* name, double* attrval);
    int admm_get_var_solution(model_t model, int var, double* sol);

    # LinAlg
    int admm_create_dense(tensor_t* tensor, int ndim, int* shape, double* data);
    int admm_create_sparse(tensor_t* tensor, int ndim, int* shape, size_t nzs, size_t* ind, double* val);
    int admm_create_from_model(tensor_t* tensor, model_t model, int isdense, int id);
    int admm_tensor_toggle_type(tensor_t tensor);
    int admm_tensor_copy(tensor_t* dest, tensor_t src);
    int admm_destroy_tensor(tensor_t tensor);

    int admm_tensor_is_dense(tensor_t tensor);
    int admm_tensor_has_attr(tensor_t tensor, int attr, int* result);
    int admm_tensor_get_shape(tensor_t tensor, int* ndim, int* shape);
    int admm_tensor_get_dense(tensor_t tensor, double* data);
    int admm_tensor_get_sparse(tensor_t tensor, size_t* nzs, size_t* ind, double* val);

    int admm_tensor_unary_op(tensor_t tensor, int opr);
    int admm_tensor_binary_op(tensor_t a, tensor_t b, int opr);
    int admm_tensor_slice_op(tensor_t tensor, int nranges, int* starts, int* stops, int* steps, tensor_t result);
    int admm_tensor_reshape_op(tensor_t tensor, int ndim, int* shape);

    #int admm_tensor_abs(tensor_t tensor);
    #int admm_tensor_exp(tensor_t tensor);
    #int admm_tensor_norm1(tensor_t tensor, double* result);
    #int admm_tensor_norm2(tensor_t tensor, double* result);
    #int admm_tensor_normf(tensor_t tensor, double* result);
    #int admm_tensor_entropy(tensor_t tensor);
    #int admm_tensor_inverse(tensor_t tensor);
    #int admm_tensor_log(tensor_t tensor);
    #int admm_tensor_square(tensor_t tensor);
    #int admm_tensor_huber(tensor_t tensor);
    #int admm_tensor_sum(tensor_t tensor, double* sum);

    #int admm_tensor_logistic(tensor_t tensor, tensor_t b);
    #int admm_tensor_max(tensor_t tensor, tensor_t b);
    #int admm_tensor_indicator(tensor_t tensor, tensor_t lb, tensor_t ub);

    int admm_tensor_unary_fun(int fun, tensor_t tensor);
    int admm_tensor_binary_fun(int fun, tensor_t tensor1, tensor_t tensor2);
    int admm_tensor_ternary_fun(int fun, tensor_t tensor1, tensor_t tensor2, tensor_t tensor3);
    int admm_tensor_variadic_fun(int fun, int nopr, tensor_t* tensors);
    
    void admm_fun_info(int* opr, int* next, int* nargs, char** shortname, char** name, char** args);
    void admm_load_tensor_attrs(const char*** attrs, int* len);

cdef extern from "numpy.h":

    ctypedef object (*dmatmul_kernel)(void* l, void* r, int row, int col);
    int numpy_ndim(object arr);
    void numpy_shape(object arr, unsigned* shape);
    void numpy_strides(object arr, unsigned* strides);
    int numpy_c_contiguous(object arr);
    int numpy_f_contiguous(object arr);
    int numpy_aligned(object arr);
    int numpy_elestride(object arr);
    int numpy_elepointer(object arr);
    int numpy_elesize(object arr);
    int numpy_alignment(object arr);
    int numpy_size(object arr);
    # if returned type is python object, increase its refcount immediately after any casting to python type
    void* numpy_get(object arr, unsigned* idx);
    object numpy_getobj(void* arr, unsigned* idx);
    void numpy_set_obj_and_incref(object arr, unsigned* idx, object obj)
    # if returned type is python object, increase its refcount immediately after any casting to python type
    void* numpy_flat(object arr, unsigned i);
    # if returned type is python object pointer, increase its refcount immediately after any casting to python type
    void* numpy_buffer(object arr);
    char* numpy_char_buffer(object arr)
    int* numpy_int_buffer(object arr)
    unsigned long* numpy_ulong_buffer(object arr)
    double* numpy_double_buffer(object arr)
    void numpy_c2f(object arr);
    void numpy_mark_c_order(object arr);
    void numpy_copy_arr_to_ptr(object arr, double* ptr);
    void numpy_copy_ptr_to_arr(double* ptr, object arr);

cdef extern from "callback.h":
    int pygillock();
    void pygilrelease(int gstate);
    void* pysavethread();
    void pyrestorethread(void* state);