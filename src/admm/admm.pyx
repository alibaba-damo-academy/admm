# cython: language_level=3
from admm cimport *
from libc.stdlib cimport malloc, free
from enum import Enum as _Enum
import numbers as _numbers
import numpy as _np
import scipy.sparse as _sp
import scipy.special as _ss
import builtins as _builtins
import functools as _functools
import threading as _threading
import traceback as _traceback
from collections import namedtuple as _namedtuple
import socket
import urllib.parse
import os
import sys

__all__ = [
    'epsilon', 'inf', "MINIMIZE", "MAXIMIZE", "REVISION", "ADMMError", "Options", "OptionConstClass", "Model", "Constr", "Constant",  "Param", "Var", "Expr", "TuningContext", "UDFBase"
]

import sys

class ThisModule(sys.__class__):

    @property
    def epsilon(self):
        return admm_get_epsilon()

    @epsilon.setter
    def epsilon(self, value):
        if value < 0:
            raise ValueError("non-negative value expected")
        admm_set_epsilon(value)

sys.modules[__name__].__class__ = ThisModule


inf = _np.inf

cdef tensorattrnames = {}

cdef loadTensorAttrs():
    cdef const char** names
    cdef int len
    global tensorattrnames
    tensorattrnames = {}
    admm_load_tensor_attrs(&names, &len)
    for i in range(len):
        tensorattrnames[str(names[i], "utf8")] = i

# load tensor attribute names
loadTensorAttrs()


cdef str BAD_ARG            = "BAD_ARG_TYPE"
cdef str OPT_NOT_FOUND      = "OPT_NOT_FOUND"
cdef str SHAPE_MISMATCHED   = "SHAPE_MISMATCHED"


cdef int OT_NOP      = 0
cdef int OT_NEG      = 1
cdef int OT_T        = 2
cdef int OT_SLICE    = 3
cdef int OT_RESHAPE  = 4
cdef int OT_ADD      = 5
cdef int OT_SUB      = 6
cdef int OT_MUL      = 7
cdef int OT_DIV      = 8
cdef int OT_MML      = 9
cdef int OT_POW      = 10

cdef OPNAMES = ["NOP", "NEG", "TRANSPOSE", "SLICE", "RESHAPE", "ADD", "SUB", "MUL", "DIV", "MATMUL", "POW"]


cdef int AT_RANGE        = 0
cdef int AT_SHAPE        = 1
cdef int AT_SPARSE       = 2
cdef int AT_DENSE        = 3
cdef int AT_PARAMETER    = 4
cdef int AT_VARIABLE     = 5
cdef int AT_EXPR         = 6


cdef callbackLock = _threading.Lock()

cdef constant(f):
    def fset(self, value):
        raise TypeError('Value is readonly')
    def fget(self):
        return f(self)
    return property(fget, fset, None, f.__doc__)

cdef class ADMMError(Exception):
    '''@APIDOC(py.ADMMError)'''
    cdef int errno_
    cdef str message_

    def __init__(self, arg1, str arg2 = None):
        message = None
        errno = -1
        if isinstance(arg1, str) or isinstance(arg1, bytes):
            message = arg1 if isinstance(arg1, str) else str(arg1, "utf8")
            errno = admm_error_code(bytes(message, "utf8"))
        else:
            errno = int(arg1)
            if arg2 is not None:
                message = arg2
            else:
                raw = admm_explain_err(errno)
                message =  str(raw, 'utf8')

        self.message_ = message
        self.errno_ = errno
        super(ADMMError, self).__init__(self.message_)

    @constant
    def errno(self):
        '''@APIDOC(py.ADMMError.errno)'''
        return self.errno_

    @constant
    def message(self):
        '''@APIDOC(py.ADMMError.message)'''
        return self.message_

cdef ensure_array_type(obj):
    if isinstance(obj, (TensorLike, _np.ndarray, _sp.spmatrix)):
        return obj
    arr = _np.array(obj)
    if arr.dtype == object:
        raise NotImplementedError
    if not _np.issubdtype(arr.dtype, _np.number) and arr.dtype != _np.bool_:
        raise TypeError("expected numeric input, got dtype {}".format(arr.dtype))
    return arr

def _start_doc_server(port = 8080):
    '''
    Start the document server at a specific port.

    Parameters
    ----------
    port : int
        Port for document web service startup.
    
    '''
    modulepath = os.path.abspath(__file__)
    pwd = os.path.dirname(modulepath)
    ROOT = os.path.join(pwd, "doc")

    if not os.path.exists(ROOT) or not os.path.isdir(ROOT):
        raise OSError(f"Directory '{ROOT}' does not exist")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass

    sock.bind(('127.0.0.1', port))

    def bad_req(client):
        client.send(b'HTTP/1.1 400 Bad Request\n\n')
        client.close()

    def not_found(client, proto):
        msg = f'{proto} 404 Not Found\n\n'
        client.send(msg.encode())
        client.close()

    def redirect(client, proto, path):
        msg = f'{proto} 301 Moved Permanently\nLocation: {path}\n\n'
        client.send(msg.encode())
        client.close()

    def guess_mime_type(file):
        ext = ''
        idx = file.rfind('.')
        if idx >= 0:
            ext = file[idx + 1 :]
        
        if ext == '': return 'application/octet-stream'
        ext = ext.lower()

        map = {
            'html': 'text/html',
            'htm': 'text/html',
            'css': 'text/css',
            'js': 'text/javascript',
            'json': 'application/json',
            'png': 'image/apng',
            'gif': 'image/gif',
            'jpg': 'image/jpg',
            'jpeg': 'image/jpeg',
            'svg': 'image/svg+xml',
            'woff': 'font/woff',
            'woff2': 'font/woff',
            'ttf': 'font/ttf',
            'otf': 'font/otf'
        }

        mtype = map.get(ext)

        return 'application/octet-stream' if mtype is None else mtype

    def resp(client, path, proto):
        mimeType = guess_mime_type(os.path.basename(path))
        with open(path, 'rb') as f:
            data = f.read()
            msg = f'{proto} 200 OK\nContent-Length:{len(data)}\n'
            msg += f'Content-Type: {mimeType}\n\n'

            msg = msg.encode() + data
            client.send(msg)
            client.close()

    sock.listen(1024)

    sys.stderr.write(f'The documentation server has been started on port {port}\n')
    sys.stderr.write(f'To access the documentation, please go to: http://127.0.0.1:{port}\n')
    sys.stderr.write(f'Press Ctrl+C to stop the current server\n')
    sys.stderr.flush()

    try:
        while True:
            client, _ = sock.accept()
            client.settimeout(30)
            req = client.recv(1024)

            if not req.startswith(b"GET /"):
                bad_req(client)
                continue

            idx = req.find(b'\n')
            if idx < 0: 
                bad_req(client)
                continue
            
            path_proto = req[4 : idx]
            idx = path_proto.rfind(b' ')
            if idx < 0:
                bad_req(client)
                continue
            
            path = path_proto[0 : idx].strip().decode()
            path = urllib.parse.unquote(path)
            proto = path_proto[idx + 1 :].strip().decode()

            if proto not in ['HTTP/1.0', 'HTTP/1.1', 'HTTP/2.0']:
                bad_req(client)
                continue
            
            if '/.' in path or '/..' in path:
                not_found(client, proto)
                continue
            
            last = path.rfind('/')
            ppos = path.rfind('?')
            if ppos >= 0 and ppos > last:
                path = path[:ppos]

            fpath = os.path.join(ROOT, path[1:])

            # Prevent symlink escape
            if not os.path.realpath(fpath).startswith(os.path.realpath(ROOT)):
                not_found(client, proto)
                continue

            if not os.path.exists(fpath):
                not_found(client, proto)
                continue
            
            if os.path.isdir(fpath):
                IDX1 = 'index.html'
                IDX2 = 'index.htm'

                idxfile1 = os.path.join(fpath, IDX1)
                idxfile2 = os.path.join(fpath, IDX2)
                if os.path.exists(idxfile1) and os.path.isfile(idxfile1):
                    redirect(client, proto, os.path.join(path, IDX1))
                    continue
                elif os.path.exists(idxfile2) and os.path.isfile(idxfile2):
                    redirect(client, proto, os.path.join(path, IDX2))
                    continue
                else:
                    not_found(client, proto)
                    continue
                
            resp(client, fpath, proto)
    except KeyboardInterrupt:
        pass

include "dynamic.pxi"
include "variadic.pxi"
include "udf.pxi"

include "tensorlike.pxi"
include "constant.pxi"
include "param.pxi"
include "var.pxi"
include "expr.pxi"
include "constr.pxi"
include "model.pxi"

def tv2d(x, p=1):
    '''@APIDOC(py.tv2d)'''
    if isinstance(p, (Var, Expr)):
        raise TypeError("constant expression is expected for argument `p`")
    if p == 1 or p == 2:
        return _binary_fun_impl(223, x, p)
    raise ValueError("unsupported p (only 1 or 2)")

# --- Bug-fix overrides (must come AFTER include "dynamic.pxi" so they shadow ---
# --- the auto-generated versions when the module loads)                       ---

def norm(x, ord=None):
    '''@APIDOC(py.norm)'''
    if ord==1: return _unary_fun_impl(103, x)
    if ord==2: return _unary_fun_impl(104, x)
    if ord=='fro': return _unary_fun_impl(105, x)
    if ord==inf: return _unary_fun_impl(116, x)
    if ord==None: return _unary_fun_impl(117, x)
    if ord=='nuc': return _unary_fun_impl(121, x)
    raise ValueError("unexpected argument(s)")

def maximum(x, b):
    '''@APIDOC(py.maximum)'''
    if isinstance(b, (Var, Expr)):
        # maximum is commutative — accept maximum(const, expr) by swapping args
        if isinstance(x, (Var, Expr)):
            raise TypeError("constant expression is expected for argument `b`")
        return _binary_fun_impl(201, b, x)
    return _binary_fun_impl(201, x, b)