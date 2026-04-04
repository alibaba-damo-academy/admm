import platform
import sys
import os
import re

cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cwd)

import ctypes
from collections import namedtuple
import template
import markdown

from .descinfo import LIBNAME, PACKAGE_VERSION
lib = None

class Option:
    def __init__(self, index, isint=True):
        global lib
        name = ctypes.c_char_p()
        writable = ctypes.c_int32()
        minn = ctypes.c_int32()
        maxn = ctypes.c_int32()
        defn = ctypes.c_int32()

        minf = ctypes.c_double()
        maxf = ctypes.c_double()
        deff = ctypes.c_double()

        
        cndoc = ctypes.c_char_p()
        endoc = ctypes.c_char_p()
        if isint:
            lib.admm_get_int_opt_info(index, ctypes.byref(name), ctypes.byref(writable), ctypes.byref(minn), 
                                      ctypes.byref(maxn), ctypes.byref(defn), ctypes.byref(cndoc), ctypes.byref(endoc))
        else:
            lib.admm_get_double_opt_info(index, ctypes.byref(name), ctypes.byref(writable), ctypes.byref(minf), 
                                         ctypes.byref(maxf), ctypes.byref(deff), ctypes.byref(cndoc), ctypes.byref(endoc))
            
        self.name = str(name.value, "utf8")
        self.min = minn.value if isint else minf.value
        self.max = maxn.value if isint else maxf.value
        self.default = defn.value if isint else deff.value
        self.writable = writable.value
        self.cndoc = str(cndoc.value, "utf8")
        self.endoc = str(endoc.value, "utf8")

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self.__dict__)

class Hyper:
    def __init__(self, index, isint=True):
        global lib
        name = ctypes.c_char_p()
        writable = ctypes.c_int32()
        minn = ctypes.c_int32()
        maxn = ctypes.c_int32()
        defn = ctypes.c_int32()

        minf = ctypes.c_double()
        maxf = ctypes.c_double()
        deff = ctypes.c_double()

        cndoc = ctypes.c_char_p()
        endoc = ctypes.c_char_p()
        if isint:
            lib.admm_get_int_hyper_info(index, ctypes.byref(name), ctypes.byref(writable), ctypes.byref(minn), 
                                        ctypes.byref(maxn), ctypes.byref(defn), ctypes.byref(cndoc), ctypes.byref(endoc))
        else:
            lib.admm_get_double_hyper_info(index, ctypes.byref(name), ctypes.byref(writable), ctypes.byref(minf), 
                                           ctypes.byref(maxf), ctypes.byref(deff), ctypes.byref(cndoc), ctypes.byref(endoc))
            
        self.name = str(name.value, "utf8")
        self.min = minn.value if isint else minf.value
        self.max = maxn.value if isint else maxf.value
        self.default = defn.value if isint else deff.value
        self.writable = writable.value
        self.cndoc = str(cndoc.value, "utf8")
        self.endoc = str(endoc.value, "utf8")

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self.__dict__)
    
class FunctionList(list):
    def __init__(self):
        global lib
        opr = ctypes.c_int32(-1)
        next = ctypes.c_int32()
        nargs = ctypes.c_int32()
        sname = ctypes.c_char_p()
        name = ctypes.c_char_p()
        args = ctypes.c_char_p()
        Func = namedtuple("Func", ["opr", "nargs", "name", "args", "const_args"])

        while True:
            lib.admm_fun_info(ctypes.byref(opr), ctypes.byref(next), ctypes.byref(nargs),
                              ctypes.byref(sname), ctypes.byref(name), ctypes.byref(args))
            
            const_args = ""
            oargs = str(args.value, "utf8")
            PREF = "const "
            pargs = []
            if oargs.strip() != "":
                for arg in oargs.strip().split(","):
                    arg = arg.strip()
                    if arg.startswith(PREF):
                        const_args += "1"
                        pargs.append(arg[len(PREF):].strip())
                    else:
                        const_args += '0'
                        pargs.append(arg)
 
            self.append(Func(opr.value, nargs.value, str(name.value, "utf8"), ",".join(pargs), const_args))

            if next.value < 0: break
            opr.value = next.value

def load(libdir):
    global lib
    if lib is None:
        if platform.system() == "Linux":
            lib = ctypes.CDLL(os.path.join(libdir, f"lib{LIBNAME}.so"))
        elif platform.system() == "Darwin":
            lib = ctypes.CDLL(os.path.join(libdir, f"lib{LIBNAME}.dylib"))
        elif platform.system() == "Windows":
            lib = ctypes.CDLL(os.path.join(libdir, f"{LIBNAME}.dll"))
        else:
            raise ValueError("Unsupported system: {}".format(platform.system()))
    nopts = []
    dopts = []

    nhyps = []
    dhyps = []

    for i in range(lib.admm_num_int_opts()):
        nopts.append(Option(i, True))

    for i in range(lib.admm_num_double_opts()):
        dopts.append(Option(i, False))

    for i in range(lib.admm_num_int_hypers()):
        nhyps.append(Hyper(i, True))

    for i in range(lib.admm_num_double_hypers()):
        dhyps.append(Hyper(i, False))

    return nopts, dopts, nhyps, dhyps, FunctionList()


def genPxiAndDoc(libdir):
    meta = load(libdir)

    tplFile = os.path.join(cwd, "..", "src", "admm", "template.pxi")
    pxiFile = os.path.join(cwd, "..", "src", "admm", "dynamic.pxi")
    docFile = os.path.join(cwd, "..", "docs", "dynamic.data")

    with open(tplFile) as i:
        with open(pxiFile, "w") as o:
            o.write(template.Template().format({"meta": meta}, i.read()))

    with open(docFile, "w") as f:
        for nopt in meta[0]:
            info = (nopt.min, nopt.max, nopt.default, 'true' if nopt.writable else 'false')
            f.write("py.OptionConstClass.{}".format(nopt.name))
            f.write("\n<! type int")
            f.write("\n<! brief")
            f.write("\n<Type: int, min: {}, max: {}, default: {}, settable: {}".format(*info))
            f.write("\n<\n<{}".format(nopt.endoc.replace("\n", "\n<")))
            f.write("\n")

        for dopt in meta[1]:
            info = (dopt.min, dopt.max, dopt.default, 'true' if dopt.writable else 'false')
            f.write("py.OptionConstClass.{}".format(dopt.name))
            f.write("\n<! type float")
            f.write("\n<! brief")
            f.write("\n<Type: float, min: {}, max: {}, default: {}, settable: {}".format(*info))
            f.write("\n<\n<{}".format(dopt.endoc.replace("\n", "\n<")))
            f.write("\n")

        for nhyper in meta[2]:
            info = (nhyper.min, nhyper.max, 'true' if nhyper.writable else 'false')
            f.write("py.TuningContext.{}".format(nhyper.name))
            f.write("\n<! type int")
            f.write("\n<! brief")
            f.write("\n<Type: int, min: {}, max: {}, settable: {}".format(*info))
            f.write("\n<\n<{}".format(nhyper.endoc.replace("\n", "\n<")))
            f.write("\n")

        for dhyper in meta[3]:
            info = (dhyper.min, dhyper.max, 'true' if dhyper.writable else 'false')
            f.write("py.TuningContext.{}".format(dhyper.name))
            f.write("\n<! type float")
            f.write("\n<! brief")
            f.write("\n<Type: float, min: {}, max: {}, settable: {}".format(*info))
            f.write("\n<\n<{}".format(dhyper.endoc.replace("\n", "\n<")))
            f.write("\n")

class _DocConverter:
    def __init__(self, doclist, name, en=True):
        self.doclist = doclist
        self.name = name
        self.en = en

    def renderBody(self, section, rst=False):
        body = section.body
        para = markdown.parseArticle(body)
        renderer = markdown.RstRenderer(para) if rst else markdown.ArticleRenderer(para, 80)
        return "\n" + renderer.render() + "\n"
    
    def pydocTitle(self, title):
        return title + "\n" + '-' * len(title)
    
    def prettyArg(self, arg):
        arg = re.subn(":([^ ])", ": \g<1>", arg)[0]
        arg = re.subn("\s*,\s*", ", ", arg)[0]
        arg = re.subn("\s*=\s*", " = ", arg)[0]
        return arg

    def toPyDoc(self):
        text = self.doclist.endoc(self.name) if self.en else self.doclist.cndoc(self.name)
        ms = markdown.MultiSection(text)
        if not ms.hasAttr("brief"):
            raise ValueError("{} has no brief".format(self.name))
        if not ms.hasAttr("type"):
            raise ValueError("{} has no type".format(self.name))
        
        doc = ""
        if ms.type.value not in ("class", "method", "function"):
            doc = "type: {}\n".format(ms.type.value)
        
        doc += self.renderBody(ms.brief)

        if ms.hasAttr("arglist"):
            doc += "\n" + self.pydocTitle("Parameters")
            for arg in ms.arglist:
                doc += "\n"
                doc += self.prettyArg(arg.value)
                doc += "\n" + self.renderBody(arg).replace("\n", "\n  ")

        if ms.hasAttr("return"):
            doc += "\n" + self.pydocTitle("Returns")
            doc += "\n" + ms["return"].value
            doc += "\n" + self.renderBody(ms["return"])

        if ms.hasAttr("note"):
            doc += "\n" + self.pydocTitle("Notes")
            doc += "\n" + self.renderBody(ms.note)

        if ms.hasAttr("example"):
            doc += "\n" + self.pydocTitle("Examples")
            doc += "\n" + self.renderBody(ms.example)

        return doc
    
PYI_HEADER = '''from typing import Union, List, Iterable, Sequence, Tuple, Any, Dict, Optional, Callable, overload
import numbers
import numpy as _np
import scipy as _sp

_ArrayLike = Union[numbers.Number, _sp.sparse.spmatrix, Iterable[Any], Sequence[Any]]'''

class PyiNode(dict):
    TAB = ' ' * 4

    def __init__(self, doclist, name):
        self.doclist = doclist
        self.path = name.strip()
        self.name = self.path.split(".")[-1].strip()
        self.depth = len(self.path.split(".")) - 2
        self.base = None
        multisec = markdown.MultiSection(self.doclist.endoc(name))

        if multisec.hasAttr("base"):
            self.base = multisec.base.value
        self.type = multisec.type.value
        self.returntype = 'None'
        self.doc = _DocConverter(doclist, name).toPyDoc().replace('\n', '\\n')
        self.args = []

        if self.isCallable() and multisec.hasAttr("arglist"):
            args = list(map(lambda arg: arg.value, multisec.arglist))
            if self.type == 'method': args = ['self'] + args
            self.args = args

        if multisec.hasAttr("return"):
            self.returntype = multisec["return"].value


    def isCallable(self):
        return self.type in ('function', 'method')

    def tab(self, num = 0):
        return PyiNode.TAB * (self.depth + num)
    
    def isOverload(self):
        if self.type != "method": return False
        if self.name == "__init__": return False
        comp = self.path.split(".")
        classpath = ".".join(comp[:-1])
        classInfo = self.doclist.endoc(classpath)
        multisec = markdown.MultiSection(classInfo)
        if not multisec.hasAttr("base"): return False
        basepath = "py." + multisec.base.value + "." + self.name

        try:
            doc = self.doclist.endoc(basepath)
            return doc is not None
        except:
            return False

    def toInterface(self, depth = 0):
        signature = ""
        if self.isCallable():
            if self.isOverload():
                signature += "{}@overload".format(self.tab())
            sig = "\n{}def {}({}) -> {}:".format(self.tab(), self.name, ", ".join(self.args), self.returntype)
            sig = sig.replace("np.ndarray", "_np.ndarray")
            sig = sig.replace("ArrayLike", "_ArrayLike")
            signature += sig
            signature += "\n{}'''{}'''".format(self.tab(1), self.doc)
            signature += "\n{}pass".format(self.tab(1))
        elif self.type == "class":
            signature += "\n{}class {}".format(self.tab(), self.name)
            if self.base is not None:
                signature += "(" + self.base + ")"
            signature += ":\n" + self.tab(1)
            signature += "'''{}'''".format(self.doc)
        else:
            type = self.type.replace("np.ndarray", "_np.ndarray")
            type = type.replace("ArrayLike", "_ArrayLike")
            signature += "\n{}{}: {}".format(self.tab(), self.name, type)
        return signature
    
class PyiGenerator:

    def __init__(self, doclist):
        self.doclist = doclist

    def generate(self):
        pyi = PYI_HEADER + "\n"
        for key in self.doclist:
            sig = PyiNode(self.doclist, key).toInterface()
            pyi += sig + "\n"

        pyi += "Options: OptionConstClass"
        return pyi

    
def fillDoc(filename, pydoc):
    dirname = os.path.dirname(filename)
    code = ""
    with open(filename, "r") as f:
        for line in f.readlines():
            m1 = re.match("include\s*([^\s]*)", line.strip())
            m2 = re.search("@APIDOC\(([^\)]*)\)", line.strip())
            if m1:
                line = fillDoc(os.path.join(dirname, m1.group(1).strip('"').strip("'")), pydoc) + "\n"
            elif m2:
                line = line.replace(m2.group(0), pydoc[m2.group(1)])
                
            code += line

    return code

def mergeAndFillDoc():
    rev = PACKAGE_VERSION

    docFile = os.path.join(cwd, "doc.data")
    auxFile = os.path.join(cwd, "dynamic.data")

    dirname = os.path.join(cwd, "..", "src", "admm")
    inFile = os.path.join(dirname, "admm.pyx")
    outFile = os.path.join(dirname, "mdcp.pyx")

    doclist = markdown.DocList([docFile, auxFile])
    pydoc = {}
    for name in doclist:
        pydoc[name] = _DocConverter(doclist, name).toPyDoc().replace("\n", "\\n")

    with open(outFile, "w") as f:
        f.write(fillDoc(inFile, pydoc))
        f.write("\nREVISION='{}'\n".format(rev))

def genPyInterface():
    docFile = os.path.join(cwd, "..", "docs", "doc.data")
    auxFile = os.path.join(cwd, "..", "docs", "dynamic.data")

    dirname = os.path.join(cwd, "..", "src", "admm")
    pyiFile = os.path.join(cwd, "..", "docs", "__init__.pyi")

    doclist = markdown.DocList([docFile, auxFile])

    with open(pyiFile, "w") as f:
        f.write(PyiGenerator(doclist).generate())
        f.write("\ninf: float")
        f.write("\nMINIMIZE: int")
        f.write("\nMAXIMIZE: int")
        

def genRstDoc():
    pass
