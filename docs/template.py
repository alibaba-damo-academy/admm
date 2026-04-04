'''
A simple template engine like java VelocityEngine.

Statement between opentag (default '{$') and closetag (default '$}') should be valid python statement.

Demo 1: Simple demo
============================================================================================================

{$
def format(n):
    word = "fisrt" if n == 1 else ("second" if n == 2 else "{}th".format(n))
    return "The {} element".format(word)
$}

{$ data = [1, 2, 3, 4] $}
{$ for i in range(len(data)) $}
    {$ format(data[i]) $}
{$ end $}


Demo 2: Use user-defined opentag & closetag
============================================================================================================

.... $$
.... $$
$$ sayhello = 'Hello' $$
$$ to = 'world' $$
$$ sayhello $$ $$ to $$!


Demo 3: Define context from python script
============================================================================================================

ctx.py:
------------------------------------------------------------------------------------------------------------
data = [1, 2, 3, 4]

tpl.txt:
------------------------------------------------------------------------------------------------------------
{$ for i in range(len(data)) $}
    The {$ if data[i] == 1 $}first{$ elif data[i] == 2 $}second{$ else $}{$ data[i] $}th{$ end $} element
{$ end $}

shell:
------------------------------------------------------------------------------------------------------------
python3 tpl.py ctx.py tpl.txt

'''
import random
import re
import os

WS = re.compile(r"\s+")
EXEC_STMT_PH = '\u0001placeholder\u0001'

class TemplateBlock:
    def name(self): return type(self).__name__
    def render(self, context): pass
    def repr(self): pass

class BlockList(TemplateBlock):
    def __init__(self, li = None):
        self.children = list(li) if li is not None else []

    def append(self, child):
        self.children.append(child)

    def render(self, context):
        ctx = context
        ctx["self"] = self
        method = "method{}".format(random.randint(0, 2 ** 32))
        ctx[method] = lambda: ''.join(map(lambda child: child.render(ctx), self.children))
        return eval("{}()".format(method), ctx)

    def repr(self):
        return {"type": self.name(), "blocklist": [child.repr() for child in self.children]}


class ConditionalBlock(TemplateBlock):
    def __init__(self):
        self.branches = []
        self.branch_bodies = []
        self.default_body = BlockList()

    def addBranch(self, expr):
        self.branches.append(expr)
        self.branch_bodies.append(BlockList())
        return self.branch_bodies[-1]

    def addDefault(self):
        return self.default_body

    def render(self, context):
        for i in range(len(self.branches)):
            expr = self.branches[i]
            bl = self.branch_bodies[i]
            if eval(expr, context):
                # if/elif + result + elif/else/end
                return EXEC_STMT_PH + bl.render(context) + EXEC_STMT_PH

        # else + result + end
        return EXEC_STMT_PH + self.default_body.render(context) + EXEC_STMT_PH

    def repr(self):
        return {
            "type": self.name(), 
            "branches": [
                {"cond": self.branches[i], "body": self.branch_bodies[i].repr() }
                for i in range(len(self.branches))
            ],
            "default": self.default_body.repr() 
        }


class LoopBlock(TemplateBlock):
    def __init__(self, loopexpr):
        self.loopexpr = loopexpr
        self.children = []

    def append(self, block):
        self.children.append(block)

    def render(self, context):
        result = "result{}".format(random.randint(0, 2 ** 32))
        ctx = context
        ctx[result] = ''
        ctx["EXEC_STMT_PH"] = EXEC_STMT_PH
        ctx["self"] = self
        method = "method{}".format(random.randint(0, 2 ** 32))
        ctx[method] = lambda: ''.join(map(lambda child: EXEC_STMT_PH + child.render(ctx), self.children))
        exec("{}:{}+={}()".format(self.loopexpr, result, method), ctx)
        # result + end
        return ctx[result] + EXEC_STMT_PH

    def repr(self):
        return {
            "type": self.name(), 
            "loopexpr": self.loopexpr,
            "body": [child.repr() for child in self.children]
        }


class EvalBlock(TemplateBlock):
    def __init__(self, text):
        self.text = text

    def render(self, context):
        try:
            res = eval(self.text, context)
            return "{}".format(res if res is not None else '')
        except:
            exec(self.text, context)
            return EXEC_STMT_PH

    def repr(self):
        return {
            "type": self.name(), 
            "expr": self.text
        }


class LiteralBlock(TemplateBlock):
    def __init__(self, content):
        self.content = content

    def render(self, context):
        return self.content
    
    def repr(self):
        return {
            "type": self.name(), 
            "content": self.content
        }

class Template:
    def __init__(self, opentag = '{$', closetag = '$}'):
        self.opentag = opentag
        self.closetag = closetag

    def tokenize(self, template):
        offset = 0
        tokens = []
        prev = 0

        lines = template.split("\n")
        if len(lines) > 0 and lines[0].strip().startswith('....'):
            self.opentag = lines[0].strip()[4:].strip()
            if len(WS.split(self.opentag)) > 1:
                raise ValueError("opentag contains whitespace(s): '{}'".format(self.opentag))
            template = template[len(lines[0]) + 1:]
            if len(lines) > 1 and lines[1].strip().startswith('....'):
                self.closetag = lines[1].strip()[4:].strip()
                if len(WS.split(self.closetag)) > 1:
                    raise ValueError("closetag contains whitespace(s): '{}'".format(self.closetag))
                template = template[len(lines[1]) + 1:]
                

        while True:
            idx1 = template.find(self.opentag, offset)
            idx2 = template.find(self.closetag, offset)
            if idx1 < 0 and idx2 < 0:
                if offset != len(template):
                    tokens.append(template[offset:])
                break
            idx1 = idx1 if idx1 >= 0 else 2**32
            idx2 = idx2 if idx2 >= 0 else 2**32
            idx = idx1 if idx1 < idx2 else idx2
            if idx != offset: tokens.append(template[offset : idx])
            if idx1 == idx2:
                # opentag == closetag
                tokens.append(0 if prev else 1)
                prev = 0 if prev else 1
            else:
                tokens.append(1 if idx1 < idx2 else 0)
            offset = idx + len(self.opentag if idx1 < idx2 else self.closetag)

        return tokens

    def compile(self, tokens):
        stack = [BlockList()]
        offset = 0

        otl = len(self.opentag)
        ctl = len(self.closetag)

        marks = []
        for token in tokens:
            if token == 1: 
                marks.append(offset)
                offset += otl
            elif token == 0:
                if len(marks) == 0:
                    raise RuntimeError("bad close tag '{}' at {}".format(self.closetag, offset))
                marks.pop()
                offset += ctl
            else:
                offset += len(token)
        if len(marks) != 0:
            raise RuntimeError("unclosed tag '{}' at {}".format(self.opentag, marks[-1]))

        state = 0
        prev = []

        offset = 0
        for token in tokens:
            if type(token) == int:
                state = token
                offset += otl if token else ctl
            elif state == 0:
                stack[-1].append(LiteralBlock(token))
                offset += len(token)
            else:
                code = token.strip()
                prefix = WS.split(code.lower())[0]
                if prefix == "if":
                    prev.append(prefix)
                    block = ConditionalBlock()
                    stack[-1].append(block)
                    stack.append(block)
                    body = block.addBranch(code[len(prefix):].strip())
                    stack.append(body)
                elif prefix == "elif":
                    if prev[-1] != "if" and prev[-1] != "elif":
                        raise RuntimeError("invalid 'elif' at {}".format(offset))
                    stack.pop()
                    body = stack[-1].addBranch(code[len(prefix):].strip())
                    stack.append(body)
                    prev.append(prefix)
                elif prefix == "else":
                    if prev[-1] != "if" and prev[-1] != "elif":
                        raise RuntimeError("invalid 'else' at {}".format(offset))
                    stack.pop()
                    body = stack[-1].addDefault()
                    stack.append(body)
                    prev.append(prefix)
                elif prefix == "for":
                    prev.append(prefix)
                    block = LoopBlock(code.strip())
                    stack[-1].append(block)
                    stack.append(block)
                elif prefix == "while":
                    prev.append(prefix)
                    block = LoopBlock(code.strip())
                    stack[-1].append(block)
                    stack.append(block)
                elif prefix == "end":
                    if prev[-1] == "for" or prev[-1] == "while":
                        prev.pop()
                        # Pop LoopBlock
                        stack.pop()
                    else:
                        # Pop branch BlockList
                        stack.pop()
                        # Pop ConditionalBlock
                        stack.pop()
                        while prev.pop() != "if": pass
                else:
                    stack[-1].append(EvalBlock(code))
                offset += len(token)
            #print(str(token) + ":" + str(list(map(lambda b: b.name(), stack))))

        return stack[0]

            
    def format(self, context, template):
        tokens = self.tokenize(template)
        block = self.compile(tokens)
        result = block.render(context)
        lines = result.split("\n")
        result = ""
        for line in lines:
            if EXEC_STMT_PH in line:
                line = line.replace(EXEC_STMT_PH, '')
                if line.strip() != "":
                    if result != "": result += "\n"
                    result += line.rstrip()
            else:
                if result != "": result += "\n"
                result += line.rstrip()

        return result

class PyModuleLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        ctx = {}
        with open(self.path, "r") as f:
            exec(f.read(), ctx, ctx)
        ctx = {key: ctx[key] for key in ctx if not key.startswith("__")}
        return ctx


class DBLoader:
    library_loaded = False

    def __init__(self, db):
        self.db = db
        if not DBLoader.library_loaded:
            root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            mod = os.path.join(root, "doc", "markdown.py")
            ctx = {}
            with open(mod, "r") as f:
                exec(f.read(), ctx, ctx)
            self.DocList = ctx["DocList"]
            DBLoader.library_loaded = True

    def load(self):
        return {"doclist": self.DocList(self.db)}

import sys

if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 4:
        sys.stderr.write("{} [CTX_FILE] TPL_FILE\n".format(sys.argv[0]))
        sys.stderr.write("example:\n")
        sys.stderr.write("\t{} ctx.py a.tpl\n".format(sys.argv[0]))
        sys.stderr.write("\t{} ../doc/doc.data a.tpl\n".format(sys.argv[0]))
        sys.exit(1)
    tpl = ''
    ctx = {}
    with open(sys.argv[-1], 'r') as f:
        tpl = f.read()
    if len(sys.argv) > 2:
        ctxf = sys.argv[1]
        if ctxf.endswith(".py"):
            ctx = PyModuleLoader(ctxf).load()
        elif ctxf.endswith(".data"):
            ctx = DBLoader(ctxf).load()
    print(Template().format(ctx, tpl))