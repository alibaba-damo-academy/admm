import re
from sphinx.roles import XRefRole
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util.nodes import make_refnode
from sphinx.util.docfields import Field, GroupedField


class CommentsRemover:
    """
    Remove all comments from source code by replacing character of comments to white space.
    So it never change the position of a specified token outside comments.
    """
    NORMAL = 0
    STR = 1
    CHAR = 2
    STR_PRE_ESCAPE = 3
    CHR_PRE_ESCAPE = 4
    PRE_COMMENTS = 5
    BLOCK_COMMENTS = 6
    LINE_COMMENTS = 7
    PRE_BC_STOP = 8

    def __init__(self, source, py=False):
        self.trans = {}
        self.defs = {}
        self.state = 0
        self.source = source
        self.py = py

        self.add(CommentsRemover.NORMAL, '"', CommentsRemover.STR)
        self.add(CommentsRemover.NORMAL, "'", CommentsRemover.CHAR)
        self.add(CommentsRemover.NORMAL, '/', CommentsRemover.PRE_COMMENTS)
        if self.py: self.add(CommentsRemover.NORMAL, '#', CommentsRemover.LINE_COMMENTS)
        self.adddef(CommentsRemover.NORMAL, CommentsRemover.NORMAL)

        self.add(CommentsRemover.STR, '"', CommentsRemover.NORMAL)
        self.add(CommentsRemover.STR, '\\', CommentsRemover.STR_PRE_ESCAPE)
        self.adddef(CommentsRemover.STR, CommentsRemover.STR)
        self.add(CommentsRemover.CHAR, "'", CommentsRemover.NORMAL)
        self.add(CommentsRemover.CHAR, '\\', CommentsRemover.CHR_PRE_ESCAPE)
        self.adddef(CommentsRemover.CHAR, CommentsRemover.CHAR)

        self.adddef(CommentsRemover.STR_PRE_ESCAPE, CommentsRemover.STR)
        self.adddef(CommentsRemover.CHR_PRE_ESCAPE, CommentsRemover.CHAR)

        self.add(CommentsRemover.PRE_COMMENTS, '/', CommentsRemover.LINE_COMMENTS)
        self.add(CommentsRemover.PRE_COMMENTS, '*', CommentsRemover.BLOCK_COMMENTS)
        self.adddef(CommentsRemover.PRE_COMMENTS, CommentsRemover.NORMAL)

        self.add(CommentsRemover.LINE_COMMENTS, '\n', CommentsRemover.NORMAL)
        self.adddef(CommentsRemover.LINE_COMMENTS, CommentsRemover.LINE_COMMENTS)

        self.add(CommentsRemover.BLOCK_COMMENTS, '*', CommentsRemover.PRE_BC_STOP)
        self.adddef(CommentsRemover.BLOCK_COMMENTS, CommentsRemover.BLOCK_COMMENTS)

        self.add(CommentsRemover.PRE_BC_STOP, '/', CommentsRemover.NORMAL)
        self.add(CommentsRemover.PRE_BC_STOP, '*', CommentsRemover.PRE_BC_STOP)
        self.adddef(CommentsRemover.PRE_BC_STOP, CommentsRemover.BLOCK_COMMENTS)

    def add(self, fromst, c, tost):
        self.trans[(fromst, c)] = tost

    def adddef(self, fromst, tost):
        self.defs[fromst] = tost

    def transform(self, c):
        key = (self.state, c)
        if key in self.trans:
            self.state = self.trans[key]
        else:
            self.state = self.defs[self.state]

    def in_comments(self, c):
        s = chr(c) if isinstance(c, int) else c
        self.transform(s)
        return self.state in (
            CommentsRemover.LINE_COMMENTS, CommentsRemover.BLOCK_COMMENTS, CommentsRemover.PRE_BC_STOP)

    def read(self):
        incomments = False
        just_leave_comments = False
        result = "" if isinstance(self.source, str) else bytearray()
        for i in range(len(self.source)):
            c = self.source[i]
            state = self.in_comments(c)
            if state != incomments and state and len(result) > 0:
                if not self.py or c != 35:
                    if isinstance(result, str):
                        result = result[:-1] + ' '
                    else:
                        result[-1] = 32

            if state or just_leave_comments:
                if isinstance(result, str):
                    if c != '\n': c = ' '
                else:
                    if c != 10: c = 32

            if isinstance(result, str):
                result += c
            else:
                result.append(c)

            just_leave_comments = self.state == CommentsRemover.PRE_BC_STOP
            incomments = state
        return result


class Token:
    MARK = "MARK"
    ID = "ID"
    KEYWORD = "KEYWORD"
    LITERAL = "LITERAL"
    PUNCT = "PUNCT"

    @staticmethod
    def match(tokens, i, openToken, closeToken):
        if i < 0 or i >= len(tokens): return -1
        if tokens[i].spelling != openToken: return -1
        depth = 1
        i += 1
        while i < len(tokens):
            if tokens[i].spelling == openToken:
                depth += 1
            elif tokens[i].spelling == closeToken:
                depth -= 1
            if depth == 0: return i
            i += 1
        return -1

    @staticmethod
    def first(tokens, occ, start=0):
        while start < len(tokens):
            if tokens[start].spelling == occ: return start
            start += 1
        return -1

    @staticmethod
    def last(tokens, occ, start=0x7fffffffffffffff):
        start = min(start, len(tokens) - 1)
        while start >= 0:
            if tokens[start].spelling == occ: return start
            start -= 1
        return -1

    @staticmethod
    def stringify(tokens):
        res = b''

        for i in range(len(tokens)):
            if i > 0 and tokens[i].kind != Token.PUNCT and tokens[i - 1].kind != Token.PUNCT:
                res += b' '
            res += bytes(tokens[i].spelling, 'utf-8')

        return res

    @staticmethod
    def debug(tokens):
        res = b''

        for i in range(len(tokens)):
            if i > 0 and tokens[i].kind != Token.PUNCT and tokens[i - 1].kind != Token.PUNCT:
                res += b' '
            res += bytes(tokens[i].kind[0] + '(' + tokens[i].spelling + ')', 'utf-8')

        return res

    def __init__(self, kind, spelling, row, col):
        self.kind = kind
        self.spelling = spelling
        self.row = row
        self.col = col
        self.attr = {}

    def __repr__(self):
        return "{}({}):[{}, {}]".format(self.kind, self.spelling, self.row, self.col)

    def __str__(self):
        return self.__repr__()


class Tokenizer:
    PUNCT = {'~', '!', '#', '%', '^', '&', '*', '(', ')', '-', '+', '=', '[', ']', '{', '}', ':', ';', ',', '<', '>',
             '.', '/', '?', '|', '\\', '@'}

    ID = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*')

    def __init__(self, source, py=False):
        source = source.encode("UTF8") if isinstance(source, str) else source
        self.source = CommentsRemover(source, py).read()
        self.py = py

    def mstr_start(self, i):
        if self.source[i] != 39: return False
        if i + 3 >= len(self.source): return False
        if self.source[i: i + 3] != b"'''": return False
        return True

    def match_mstr(self, i):
        if not self.mstr_start(i): raise ValueError("No multi-line string")
        token = bytearray()
        token += b"'''"
        pre_escape = False

        s = 0
        k = i + 3

        while k < len(self.source):
            b = self.source[k]
            token.append(b)
            if not pre_escape and b == 39:
                s += 1
                if s == 3: return token, k
            if self.source[k] == 92 and not pre_escape:
                pre_escape = True
            else:
                pre_escape = False
            k += 1

        raise ValueError("Unclosed mstring")

    def match_str(self, i):
        if self.source[i] != 34: raise ValueError("No string")
        token = bytearray()
        token.append(34)
        k = i + 1
        pre_escape = False

        while k < len(self.source):
            b = self.source[k]
            token.append(b)
            if not pre_escape and b == 34: return token, k
            if self.source[k] == 92 and not pre_escape:
                pre_escape = True
            else:
                pre_escape = False
            k += 1

        raise ValueError("Unclosed string")

    def match_chr(self, i):
        if self.source[i] != 39: raise ValueError("No char")
        token = bytearray()
        token.append(39)
        k = i + 1
        pre_escape = False

        while k < len(self.source):
            b = self.source[k]
            token.append(b)
            if not pre_escape and b == 39: return token, k
            if self.source[k] == 92 and not pre_escape:
                pre_escape = True
            else:
                pre_escape = False
            k += 1

        raise ValueError("Unclosed char")

    def isnumeric(self, token):
        spelling = token.spelling
        if len(spelling) == 0: return False
        spelling = spelling.lstrip('-')
        spelling = spelling.lstrip('.')
        return len(spelling) > 0 and spelling[0].isdigit()

    def merge_numeric_literal(self, tokens):
        tmp = []
        result = []

        # merge tokens around '.'
        for token in tokens:
            if token.spelling == '.':
                if len(result) > 0 and self.isnumeric(result[-1]):
                    result[-1].spelling += token.spelling
                else:
                    result.append(token)

            elif self.isnumeric(token):
                if len(result) > 0:
                    if self.isnumeric(result[-1]):
                        result[-1].spelling += token.spelling
                    elif result[-1].spelling == '.':
                        result[-1].spelling += token.spelling
                    else:
                        result.append(token)
                else:
                    result.append(token)
            else:
                result.append(token)

        tmp = result
        result = []

        # merge '-' and number
        for token in tmp:
            if self.isnumeric(token) and len(result) > 0 and result[-1].spelling == '-':
                if any([
                    len(result) == 1,
                    result[-2].kind == Token.PUNCT,
                    self.isnumeric(result[-2]) and result[-2].spelling.lower().endswith('e')
                ]):
                    result[-1].spelling += token.spelling

                else:
                    result.append(token)
            else:
                result.append(token)

        i = 0
        tmp = result
        result = []

        # merge [:digit:]e and its following number
        for token in tmp:
            if self.isnumeric(token) and len(result) > 0 and self.isnumeric(result[-1]):
                result[-1].spelling += token.spelling
            else:
                result.append(token)

        # mark numeric token as LITERAL
        for token in result:
            if self.isnumeric(token): token.kind = Token.LITERAL

        return result

    def tokenize(self):
        tokens = []
        token = [bytearray(), 0, 0]
        row = 1
        col = 0
        i = 0

        while i < len(self.source):
            b = self.source[i]
            p = self.source[i - 1] if i > 1 else None
            if b == 10 and p != '\\':
                row += 1
                col = 0
            else:
                col += 1
            if b == 34 or b == 39:
                if len(token[0]) > 0: tokens.append(token)

                try:
                    tk, i = self.match_str(i) if b == 34 else (
                        self.match_mstr(i) if self.py and self.mstr_start(i) else self.match_chr(i))
                except:
                    raise ValueError("Unclosed string at {}:{}".format(row, col))

                tokens.append([tk, row, col])
                col += len(tk)
                token = [bytearray(), 0, 0]
                i += 1
                continue

            if chr(b).isspace():
                if len(token[0]) > 0:
                    tokens.append(token)
                    token = [bytearray(), 0, 0]
                i += 1
                continue

            if chr(b) in Tokenizer.PUNCT:
                if len(token[0]) > 0:
                    tokens.append(token)
                    token = [bytearray(), 0, 0]
                if chr(b) != '\\': tokens.append([bytearray(chr(b), 'UTF8'), row, col])
                i += 1
                continue

            i += 1
            if len(token[0]) == 0:
                token[1] = row
                token[2] = col
            token[0].append(b)

        if len(token[0]) > 0: tokens.append(token)

        result = []
        for token in tokens:
            t = Token(None, token[0].decode('UTF8'), token[1], token[2])
            if t.spelling in Tokenizer.PUNCT:
                t.kind = Token.PUNCT
            elif Tokenizer.ID.fullmatch(t.spelling):
                t.kind = Token.ID
            else:
                t.kind = Token.LITERAL
            result.append(t)

        return self.merge_numeric_literal(result)


class FunctionSignature:
    def __init__(self, signature):
        self.args = []
        tokens = Tokenizer(signature).tokenize()

        start = Token.first(tokens, '(')
        self.name = b''
        self.restype = b''

        if not tokens: return

        if start < 0:
            self.parseNameAndResType(tokens)
            return

        if start != 0: self.parseNameAndResType(tokens[: start])

        stop = Token.match(tokens, start, '(', ')')
        if stop < 0: return

        args = tokens[start + 1: stop]
        i = 0

        start = 0
        while i < len(args):
            t = args[i]

            if t.spelling == '(':
                i = Token.match(args, i, '(', ')')
                if i < 0: break
            elif t.spelling == '<':
                i = Token.match(args, i, '<', '>')
                if i < 0: break
            elif t.spelling == '[':
                i = Token.match(args, i, '[', ']')
                if i < 0: break
            elif t.spelling == '{':
                i = Token.match(args, i, '{', '}')
                if i < 0: break
            elif t.spelling == ',':
                self.parseArg(args[start: i])
                start = i + 1

            i += 1

        if start < len(args):
            self.parseArg(args[start:])

    def parseVarDecl(self, tokens):
        i = 0
        indices = []
        while i < len(tokens):
            token = tokens[i]
            if i > 0:
                prevstop = tokens[i - 1].col + len(tokens[i - 1].spelling)
                if prevstop < token.col:
                    indices.append(i)
            if token.spelling == '<':
                i = Token.match(tokens, i, '<', '>')
            elif token.spelling == '(':
                i = Token.match(tokens, i, '(', ')')
            elif token.spelling == '[':
                i = Token.match(tokens, i, '[', ']')
            elif token.spelling == '{':
                i = Token.match(tokens, i, '{', '}')
            if i == -1: break
            i += 1

        if not indices:
            return b'', Token.stringify(tokens)

        return Token.stringify(tokens[:indices[-1]]), Token.stringify(tokens[indices[-1]:])

    def parseNameAndResType(self, tokens):
        # MDO.Indicator Model.getGenConstrIndicaror(...)
        # std::map<int, int>::iterator std::map<int, int>::begin()
        # int operator+(...)
        # double[] foo()
        self.restype, self.name = self.parseVarDecl(tokens)

    def parseArg(self, tokens):
        start = Token.first(tokens, '(')

        if start >= 0 and start + 3 < len(tokens):
            typelist = tokens[: start + 2] + tokens[start + 3:]
            namelist = tokens[start + 2: start + 3]
            self.args.append((
                Token.stringify(typelist),
                Token.stringify(namelist)
            ))
        else:
            if len(tokens) > 2 and tokens[-2].spelling == '=':
                self.args.append((
                    Token.stringify(tokens[:-3]),
                    Token.stringify(tokens[-3:])
                ))
            else:
                self.args.append(self.parseVarDecl(tokens))

    def signature(self):
        sigstr = bytearray()
        sigstr += self.name
        pair2strfn = lambda arg: arg[0] if arg[0] else arg[1]

        sigstr += b'('
        sigstr += b', '.join(map(pair2strfn, self.args))
        sigstr += b')'

        return str(sigstr, 'utf-8')


class LanguageObject(ObjectDescription):
    doc_field_types = [
        GroupedField(
            'parameter',
            label=_('Parameters'),
            names=('param', 'parameter', 'arg', 'argument'),
            can_collapse=True
        ),
        GroupedField(
            'exceptions',
            label=_('Throws'),
            rolename='expr',
            names=('throws', 'throw', 'exception'),
            can_collapse=True
        ),
        GroupedField(
            'retval',
            label=_('Return values'),
            names=('retvals', 'retval'),
            can_collapse=True
        ),
        Field(
            'returnvalue',
            label=_('Returns'),
            has_arg=False,
            names=('returns', 'return')
        )
    ]

    # static properties to override
    namespace_delim = '.'
    domain_name = 'unknown'

    type_tag = 'class'
    func_tag = 'function'
    prop_tag = 'property'

    def isKeyword(self, word):
        return False

    def isKeywordType(self, word):
        return False

    def handle_type(self, sig, signode):
        tokens = Tokenizer(sig).tokenize()
        typenode = addnodes.desc_classname()

        if not tokens: return
        start = tokens[0].col

        for token in tokens:
            if token.col != start: typenode += addnodes.desc_sig_space()
            if token.kind == Token.PUNCT:
                typenode += addnodes.desc_sig_punctuation(text=token.spelling)
            elif token.kind == Token.LITERAL:
                if token.spelling[0] == "'":
                    typenode += addnodes.desc_sig_literal_char(text=token.spelling)
                elif token.spelling[0] == '"':
                    typenode += addnodes.desc_sig_literal_string(text=token.spelling)
                else:
                    typenode += addnodes.desc_sig_literal_number(text=token.spelling)
            elif token.kind == Token.ID:
                pnode = addnodes.pending_xref(
                    '', refdomain=self.domain_name,
                    reftype='type',
                    reftarget="..." + token.spelling, modname=None,
                    classname=None
                )
                pnode += addnodes.desc_name(text=token.spelling)
                typenode += pnode
            start = token.col + len(token.spelling)

        signode += typenode

        return sig

    def handle_function(self, sig, signode):
        func = FunctionSignature(sig)

        if func.restype != b'':
            self.handle_type(str(func.restype, 'utf-8'), signode)
            signode += addnodes.desc_sig_space()
        if func.name != b'': signode += addnodes.desc_sig_name(text=str(func.name, 'utf-8'))
        signode += addnodes.desc_sig_name(text='(')

        for i in range(len(func.args)):
            arg = func.args[i]
            if i != 0:
                signode += addnodes.desc_sig_punctuation(text=',')
                signode += addnodes.desc_sig_space()
            if arg[0] != b'': self.handle_type(str(arg[0], 'utf-8'), signode)
            if arg[0] != b'' and arg[1] != b'': signode += addnodes.desc_sig_space()
            if arg[1] != b'': signode += addnodes.desc_sig_name(text=str(arg[1], "utf-8"))

        signode += addnodes.desc_sig_name(text=')')

        return sig

    def handle_property(self, sig, signode):
        return self.handle_type(sig, signode)

    def handle_signature(self, sig, signode):
        if self.objtype == self.type_tag:
            return self.handle_type(sig, signode)
        elif self.objtype == self.func_tag:
            return self.handle_function(sig, signode)
        elif self.objtype == self.prop_tag:
            return self.handle_property(sig, signode)
        else:
            raise RuntimeError("Unknown objtype: {}".format(self.objtype))

    def before_content(self):
        if "path" not in self.env.temp_data:
            self.env.temp_data["path"] = []
        self.env.temp_data["path"].append(self.names[0])

    def after_content(self):
        self.env.temp_data["path"].pop()

    def add_target_and_index(self, name_cls, sig, signode):
        if self.objtype == self.type_tag:
            signode['ids'].append(self.domain_name + 't-' + sig)
            self.env.get_domain(self.domain_name).add_type(sig)

        elif self.objtype == self.func_tag:
            func = FunctionSignature(sig)
            classname = self.env.temp_data["path"][-1]

            func.name = bytes(classname + self.namespace_delim, 'utf-8') + func.name
            funcsig = func.signature()
            signode['ids'].append(self.domain_name + 'f-' + funcsig)
            self.env.get_domain(self.domain_name).add_function(funcsig)

        elif self.objtype == self.prop_tag:
            signode['ids'].append(self.domain_name + 'p-' + sig)
            self.env.get_domain(self.domain_name).add_property(sig)


class LanguageDomain(Domain):
    name = None
    namespace_delim = '.'

    func_role = 'func'
    type_role = 'type'
    prop_role = 'prop'

    initial_data = {
        'types': [],
        'functions': [],
        'props': []
    }

    # static function to override
    @staticmethod
    def directive(name):
        return NotImplemented

    @staticmethod
    def role(name):
        return XRefRole()

    def add_member(self, sig, key, typename):
        prefix = self.name + typename
        name = prefix + '-' + sig
        self.data[key].append((name, prefix, self.env.docname, name, 0))

    def add_type(self, sig):
        self.add_member(sig, 'types', 't')

    def add_function(self, sig):
        self.add_member(sig, 'functions', 'f')

    def add_property(self, sig):
        self.add_member(sig, 'props', 'p')

    def sameArgs(self, l, r):
        if len(l.args) != len(r.args): return False
        for i in range(len(l.args)):
            type1 = l.args[i][0] if l.args[i][0] != b'' else l.args[i][1]
            type2 = r.args[i][0] if r.args[i][0] != b'' else r.args[i][1]
            if type1 != type2:
                return False
        return True

    def resolve_xref(self, env, fromdocname, builder, typename, target, node, contnode):
        typepref = self.name + 't-'
        funcpref = self.name + 'f-'
        proppref = self.name + 'p-'

        if typename == self.func_role:
            refnodes = []
            for name, tp, todoc, anchor, pri in self.data["functions"]:
                func = FunctionSignature(anchor[len(funcpref):])
                funcid = func.signature()
                trgt = FunctionSignature(target)

                refname = funcpref + funcid

                sameargs = self.sameArgs(func, trgt)
                samedoc = todoc == fromdocname
                samename = func.name == trgt.name
                bdelim = self.namespace_delim.encode('utf-8')
                realname = func.name.split(bdelim)[-1]
                samesimplename = realname == trgt.name

                if sameargs:
                    if samename or (samesimplename and samedoc):
                        return make_refnode(
                            builder, fromdocname, todoc,
                            refname, contnode, target
                        )
                elif samename or (samesimplename and samedoc):
                    refnodes.append(make_refnode(
                        builder, fromdocname, todoc,
                        refname, contnode, target
                    ))

                # if samesimplename and samedoc:
                #    print("Failed compare", funcid, anchor[len(funcpref):], target, todoc, fromdocname, sameargs)

            if len(refnodes) == 1:
                return refnodes[0]
            else:
                print('Target function {}, {} candicates'.format(target, len(refnodes)))

        elif typename == self.type_role:
            weakRef = False
            if target.startswith('...'):
                weakRef = True
                target = target[3:]
            for name, tp, todoc, anchor, pri in self.data['types']:
                if anchor == tp + '-' + target:
                    return make_refnode(
                        builder, fromdocname, todoc,
                        typepref + target, contnode, target
                    )
            if not weakRef: print('Target type {}, found nothing'.format(target))

        elif typename == self.prop_role:
            for name, tp, todoc, anchor, pri in self.data['props']:
                if anchor == tp + '-' + target:
                    return make_refnode(
                        builder, fromdocname, todoc,
                        proppref + target, contnode, target
                    )
            print('Target prop {}, found nothing'.format(target))
        else:
            print('Target {}, found nothing'.format(target))
            return None


class Language:
    def __init__(self, app, langname, namespace_sep='.'):
        class LangObj(LanguageObject):
            namespace_delim = namespace_sep
            domain_name = langname

        class LangDomain(LanguageDomain):
            namespace_delim = namespace_sep
            name = langname

            directives = {
                'function': LangObj,
                'class': LangObj,
                'property': LangObj
            }

            roles = {
                'func': XRefRole(),
                'type': XRefRole(),
                'prop': XRefRole()
            }

            @staticmethod
            def directive(name):
                return LangObj

            def merge_domaindata(self, docnames, otherdata):
                """Merge domain data from parallel builds.

                This method is required for Sphinx parallel builds.
                It merges the domain data (types, functions, props) from
                different parallel build processes.
                """
                for key in ['types', 'functions', 'props']:
                    # Merge the lists, avoiding duplicates
                    existing_names = {item[0] for item in self.data[key]}
                    for item in otherdata[key]:
                        if item[0] not in existing_names:
                            self.data[key].append(item)

        app.add_domain(LangDomain)


def setup(app):
    Language(app, 'java')
    Language(app, 'cs')

    return {
        'version': '0.2',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
