"""
RST Documentation Generator for MindOpt Python SDK

Generates reStructuredText documentation from doc.data source file.
English only version for open-source release.
"""
import os
import sys
import shutil
import re

cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cwd)

import markdown

docFile = os.path.join(cwd, "doc.data")
auxFile = os.path.join(cwd, "dynamic.data")

TITLE_PROP = "Properties"
TITLE_METHOD = "Methods"
TITLE_FUN = "Functions"
TITLE_TYPE = "Type"

def formatFun(name, fun, entityname, indent, top_level=False):
    """Format a function/method for RST output."""
    arglist = []
    if fun.hasAttr("arglist"):
        arglist += map(lambda arg: arg.value, fun.arglist)
    decl_indent = 0 if top_level else 4
    body_indent = 4 if top_level else 8
    detail_indent = 8 if top_level else 12
    subdetail_indent = 12 if top_level else 16
    tab = lambda i: ' ' * (indent + i)
    arglist = ", ".join(arglist)
    content = "{}.. py:{}:: {}({})".format(tab(decl_indent), entityname, name.split(".")[-1], arglist)
    paragraph = markdown.parseArticle(fun.brief.body)
    brief = ("\n" + markdown.RstRenderer(paragraph).render()).replace("\n", "\n" + tab(body_indent))

    if (fun.hasAttr("example")):
        examplep = markdown.parseArticle(fun.example.body)
        example = ("\n" + markdown.RstRenderer(examplep).render()).replace("\n", "\n" + tab(body_indent))
        brief += "\n\n" + example
    content += "\n" + brief + "\n\n"

    if fun.hasAttr("arglist"):
        for arg in fun.arglist:
            argp = markdown.parseArticle(arg.body)
            argbrief = ("\n" + markdown.RstRenderer(argp).render()).replace("\n", "\n" + tab(subdetail_indent))
            argname = arg.value.split(":")[0]
            argtype = arg.value.split(":", 1)[1]
            argtype = re.subn("\s*,\s*", ", ", argtype)[0]
            argtype = re.subn("\s*=\s*", " = ", argtype)[0]

            content += "{}:param {}:\n\n".format(tab(detail_indent), argname)
            argbrief = "{}{}:{}\n\n".format(tab(subdetail_indent), TITLE_TYPE, argtype) + argbrief
            content += argbrief + "\n\n"

    if fun.hasAttr("return"):
        retsec = fun["return"]
        retp = markdown.parseArticle(retsec.body)
        rettype = retsec.value
        rettype = re.subn("\s*,\s*", ", ", rettype)[0]
        rettype = re.subn("\s*=\s*", " = ", rettype)[0]
        paragraph = "{}: {}\n\n".format(TITLE_TYPE, rettype)
        retp = markdown.parseArticle(paragraph + retsec.body)
        retb = ("\n" + markdown.RstRenderer(retp).render()).replace("\n", "\n" + tab(subdetail_indent))
        content += "{}:return:\n\n".format(tab(detail_indent))
        content += retb + "\n\n"

    if fun.hasAttr("note"):
        content += "{}.. note::".format(tab(detail_indent))
        paragraph = markdown.parseArticle(fun.note.body)
        content += ("\n" + markdown.RstRenderer(paragraph).render()).replace("\n", "\n" + tab(subdetail_indent))
        content += "\n"

    return content


class ModuleConverter:

    def __init__(self, modulename, indent=0):
        self.module = modulename
        self.node = None
        self.propnames = []
        self.props = []
        self.methodnames = []
        self.methods = []
        self.indent = indent
        self.extactMembers()

    def tab(self, i=0):
        return ' ' * (self.indent + i)

    def extactMembers(self):
        doclist = markdown.DocList((docFile, auxFile))
        self.node = markdown.MultiSection(doclist.endoc(self.module))
        
        docs = []
        for key in doclist:
            if not key.startswith(self.module + "."): continue
            ms = markdown.MultiSection(doclist.endoc(key))
            if ms.type.value == "class":
                raise TypeError("Embedded class not support")
            
            sortname = key
            if key.endswith('__init__'):
                sortname = "\0"
            elif ms.hasAttr("sortname"):
                sortname = ms.sortname.value
            docs.append((sortname + '\0' + key, key, ms))
                
        docs = sorted(docs, key=lambda x: x[0])

        for _, key, ms in docs:
            if ms.type.value == "method":
                self.methodnames.append(key)
                self.methods.append(ms)
            else:
                self.propnames.append(key)
                self.props.append(ms)

    def writeModuleName(self):
        name = self.tab() + self.module.split(".")[-1].strip()
        underscore = self.tab() + '-' * len(name)
        classdecl = self.tab() + ".. py:class:: " + name
        if self.node.hasAttr("base"):
            classdecl += "({})".format(self.node.base.value)
        paragraph = markdown.parseArticle(self.node.brief.body)
        brief = ('\n' + markdown.RstRenderer(paragraph).render()).replace('\n', '\n' + self.tab(4))
        return "{}\n{}\n\n{}\n\n{}\n\n".format(name, underscore, classdecl, brief)
    
    def listProps(self):
        content = "{}**{}**".format(self.tab(4), TITLE_PROP)
        content += "\n\n{}.. list-table::".format(self.tab(4))
        content += "\n{}:widths: 15 30\n{}:class: longtable\n\n".format(self.tab(8), self.tab(8))
        for i in range(len(self.propnames)):
            name = self.propnames[i]
            prop = self.props[i]

            # Safely extract brief text with type checking
            paragraphs = markdown.parseArticle(prop.brief.body)
            valid_paras = [p for p in paragraphs if isinstance(p, markdown.Paragraph)]
            brief = valid_paras[0].content if valid_paras else ""

            if "TuningContext" in self.module or "OptionConstClass" in self.module:
                brief = valid_paras[1].content if len(valid_paras) > 1 else brief

            content += "{}* - :py:attr:`{}`\n".format(self.tab(8), name.split(".")[-1])
            content += "{}  - {}\n".format(self.tab(8), brief)
        
        return content + "\n\n"
    
    def listMethods(self):
        content = "{}**{}**".format(self.tab(4), TITLE_METHOD)
        content += "\n\n{}.. list-table::".format(self.tab(4))
        content += "\n{}:widths: 15 30\n{}:class: longtable\n\n".format(self.tab(8), self.tab(8))
        for i in range(len(self.methodnames)):
            name = self.methodnames[i]
            method = self.methods[i]
            
            # Safely extract brief text with type checking
            paragraphs = markdown.parseArticle(method.brief.body)
            valid_paras = [p for p in paragraphs if isinstance(p, markdown.Paragraph)]
            brief = valid_paras[0].content if valid_paras else ""
            
            content += "{}* - :py:meth:`{}`\n".format(self.tab(8), name.split(".")[-1])
            content += "{}  - {}\n".format(self.tab(8), brief)
        
        return content + "\n\n"
    
    def writeProp(self, name, prop):
        content = "{}.. py:property:: {}".format(self.tab(4), name.split(".")[-1])
        content += "\n{}:type: {}".format(self.tab(8), prop.type.value)
        paragraph = markdown.parseArticle(prop.brief.body)
        brief = "\n" + markdown.RstRenderer(paragraph).render()
        brief = brief.replace("\n", "\n" + self.tab(8))
        content += "\n" + brief + "\n\n"
        return content
    
    def writeProps(self):
        content = ""
        for i in range(len(self.propnames)):
            content += self.writeProp(self.propnames[i], self.props[i])
        return content + "\n\n"
    
    def writeMethod(self, name, method):
        return formatFun(name, method, "method", self.indent)

    def writeMethods(self):
        content = ""
        for i in range(len(self.methodnames)):
            content += self.writeMethod(self.methodnames[i], self.methods[i])
        return content + "\n\n"
    
    def genRst(self):
        content = self.writeModuleName()
        if len(self.propnames) > 0:
            content += self.listProps()
        if len(self.methodnames) > 0:
            content += self.listMethods()
        content += self.writeProps()
        content += self.writeMethods()
        return content


class RstRewriter:
    def __init__(self, filename, indent=0, include_all_functions=False):
        self.filename = filename
        self.content = ""
        self.funnames = []
        self.funs = []
        self.indent = indent
        self.include_all_functions = include_all_functions
        self.doclist = markdown.DocList((docFile, auxFile))
        inModule = False
        with open(filename, "r") as f:
            for line in f.readlines():
                # Filter out Chinese lines (> prefix) and remove < prefix
                if line.startswith('>'):
                    continue  # Skip Chinese lines
                if line.startswith('<'):
                    self.content += line[1:]  # Remove < prefix
                else:
                    self.content += line
                # Skip .. module:: directive
                if line.strip() == ".. module::":
                    inModule = True
                    continue
                elif inModule:
                    name = line.strip()
                    self.funnames.append(name)
                    doc = self.doclist.endoc(name)
                    self.funs.append(markdown.MultiSection(doc))

        if self.include_all_functions:
            self._collect_all_functions()

    def tab(self, i=0):
        return ' ' * (self.indent + i)

    def listFun(self):
        content = "{}**{}**".format(self.tab(4), TITLE_FUN)
        content += "\n\n{}.. list-table::".format(self.tab(4))
        content += "\n{}:widths: 15 30\n{}:class: longtable\n\n".format(self.tab(8), self.tab(8))
        for i in range(len(self.funnames)):
            name = self.funnames[i]
            method = self.funs[i]
            
            # Safely extract brief text with type checking
            paragraphs = markdown.parseArticle(method.brief.body)
            valid_paras = [p for p in paragraphs if isinstance(p, markdown.Paragraph)]
            brief = valid_paras[0].content if valid_paras else ""
            
            content += "{}* - :py:func:`{}`\n".format(self.tab(8), name.split(".")[-1])
            content += "{}  - {}\n".format(self.tab(8), brief)
        return content + "\n\n"

    def _collect_all_functions(self):
        """Collect all py.* functions from doclist, appending any not already listed."""
        seen = set(self.funnames)
        docs = []
        for key in self.doclist:
            if not key.startswith("py."):
                continue
            doc = self.doclist.endoc(key)
            if doc is None:
                continue
            ms = markdown.MultiSection(doc)
            if ms.type.value != "function":
                continue
            if key in seen:
                continue
            docs.append((key.split(".")[-1], key, ms))

        for _, key, ms in sorted(docs, key=lambda x: x[0]):
            self.funnames.append(key)
            self.funs.append(ms)


    def writeFun(self, name, fun):
        return formatFun(name, fun, "function", self.indent, top_level=True)
    
    def genRst(self):
        content = self.content
        if self.funnames:
            if not content.endswith("\n\n"):
                content += "\n" if content.endswith("\n") else "\n\n"
            if not self.include_all_functions:
                content += self.listFun()
            for name, fun in zip(self.funnames, self.funs):
                content += self.writeFun(name, fun)
        return content


def writeRst():
    """Generate RST documentation from doc.data (English only)."""
    srcRstDir = os.path.join(cwd, "sdk")
    dstRstDir = os.path.join(cwd, "5_API_Document")
    srcIdxRst = os.path.join(srcRstDir, "index.rst")
    dstIdxRst = os.path.join(dstRstDir, "index.rst")

    if os.path.exists(dstRstDir):
        shutil.rmtree(dstRstDir)
    os.makedirs(dstRstDir)

    inModule = False
    inOptions = False
    output_lines = []

    with open(srcIdxRst, "r") as i:
        for line in i.readlines():
            stripped = line.strip()
            
            if stripped == ".. toctree::":
                output_lines.append(line)
                inModule = True
                inOptions = True
            elif inModule and inOptions and stripped.startswith(":"):
                # toctree options like :maxdepth: 1
                output_lines.append(line)
            elif inModule and stripped == "":
                # Blank line ends options section
                inOptions = False
                output_lines.append(line)
            elif inModule and not inOptions and stripped and not stripped.startswith(".."):
                # Module or file entry
                name = stripped
                if not name.endswith(".rst"):
                    # Module name like py.ADMMError -> ADMMError.rst
                    render = ModuleConverter(name)
                    output_name = name.split(".")[-1] + ".rst"
                else:
                    # Already a .rst file like globals.rst
                    filename = os.path.join(srcRstDir, name)
                    # globals.rst has a hand-written summary table; collect all
                    # py.* function docs without generating a duplicate table.
                    include_all = (name == "globals.rst")
                    render = RstRewriter(filename, include_all_functions=include_all)
                    output_name = name

                with open(os.path.join(dstRstDir, output_name), "w") as m:
                    m.write(render.genRst())

                output_lines.append("    " + output_name + "\n")
            elif inModule and stripped.startswith(".."):
                # Another directive, end of toctree
                inModule = False
                output_lines.append(line)
            else:
                # Other content
                if inModule:
                    inModule = False
                output_lines.append(line)

    with open(dstIdxRst, "w") as o:
        o.writelines(output_lines)
