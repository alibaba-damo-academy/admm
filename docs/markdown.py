import re

WS = re.compile(r'\s+')
CITE = re.compile(r"(`[^`]+`)")

class Paragraph:
    def __init__(self, line):
        self.indent = len(line) - len(line.lstrip(' '))
        self.cindent = self.indent
        self.content = line.lstrip(' ')
        self.ordered = None
        self.language = None

        if self.content.startswith("* "):
            self.ordered = False
            content = self.content[1:]
            ind = len(content) - len(content.lstrip(' '))
            self.content = content.strip(' ')
            self.cindent = self.indent + 1 + ind
        elif self.content.startswith("#. "):
            self.ordered = True
            content = self.content[2:]
            ind = len(content) - len(content.lstrip(' '))
            self.content = content.strip(' ')
            self.cindent = self.indent + 2 + ind

    def append(self, content):
        self.content += " "
        self.content += content.strip(' ')

    def __str__(self):
        prefix = "p"
        if self.ordered is not None:
            prefix = "o" if self.ordered else "u"
        
        return "{}{}({}, {}){}".format(self.indent * ' ', prefix, self.indent, self.cindent, self.content.replace('\n', '\\n'))

    def __repr__(self):
        return self.__str__()


def parseArticle(text):
    paragraphs = []
    prev_ind = -1
    language = None
    code = ""

    for line in text.split("\n"):
        if language is not None:
            if line.strip().startswith("```"):
                p = Paragraph(code)
                p.language = language if language != "" else None
                language = None
                paragraphs.append(p)
                prev_ind = -1
            else:
                if code != "":
                    code += "\n"
                code += line
        elif line.strip() == "":
            if prev_ind < 0:
                paragraphs.append("\n")
            prev_ind = -1
        elif line.strip().startswith("```"):
            language = line.strip()[3:].strip()
            code = ""
        else:
            p = Paragraph(line)
            if prev_ind < 0 or p.ordered is not None or p.indent != prev_ind:
                paragraphs.append(p)
                prev_ind = p.cindent
            else:
                paragraphs[-1].append(line)

    stack = []

    for p in paragraphs:
        if not isinstance(p, str):
            while len(stack) > 0 and stack[-1][0] < p.indent:
                stack.pop()
            if p.ordered is not None and p.ordered:
                if len(stack) > 0 and stack[-1][0] == p.indent:
                    p.id = stack[-1][1]
                    stack[-1][1] += 1
                    stack[-1][2][0] += 1
                else:
                    p.id = 1
                    stack.append([p.indent, 2, [1]])
                p.ids = stack[-1][2]
    for p in paragraphs:
        if not isinstance(p, str) and p.ordered is not None and p.ordered:
            p.ids = p.ids[0]

    return paragraphs


class ArticleRenderer:
    def __init__(self, paragraphs, linewidth = -1):
        self.paragraphs = paragraphs
        self.linewidth = linewidth

    def _first(self, s, length):
        if len(s) <= length + 3: return s, None
        if s[length].isspace(): return s[:length], s[length + 1:]
        tol = length >> 2
        for i in range(1, tol):
            if s[length - i].isspace(): return s[:length - i], s[length - i + 1:]
            if length + i < len(s) and s[length + i].isspace(): return s[:length + i], s[length + i + 1:]
        
        if s[length - 1].isalpha() and s[length].isalpha():
            return s[:length] + '-', s[length:]
        return s[:length], s[length:]

    def breakline(self, s, indent):
        length = self.linewidth - indent
        lines = []
        rest = s

        while rest is not None:
            first, rest = self._first(rest, length)
            lines.append(first)

        return lines

    def emptyline(self):
        return ""

    def code(self, indent, content, language):
        content = "\n" + content
        content = content.replace('\n', '\n\n' + (indent + 2) * ' ')
        return content

    def ordered(self, indent, id, numids, content):
        s = str(numids)
        w = len(s)
        id = str(id)
        id = ' ' * (w - len(id)) + id
        prefix = indent * ' ' + id + '. '

        if self.linewidth > 0:
            result = prefix
            li = self.breakline(content, indent)
            for i in range(0, len(li)):
                if i != 0: result += "\n" + ' ' * len(prefix)
                result += li[i]
            return result

        return prefix + content

    def unordered(self, indent, content):
        prefix = indent * ' ' + "* "

        if self.linewidth > 0:
            result = prefix
            li = self.breakline(content, indent)
            for i in range(0, len(li)):
                if i != 0: result += "\n" + ' ' * len(prefix)
                result += li[i]
            return result

        return prefix + content

    def plain(self, indent, content):
        prefix = indent * ' '

        if self.linewidth > 0:
            result = prefix
            li = self.breakline(content, indent)
            for i in range(0, len(li)):
                if i != 0: result += "\n" + ' ' * indent
                result += li[i]
            return result

        return prefix + content

    def render(self):
        result = ""
        for p in self.paragraphs:
            if result != "": result += "\n\n"
            if isinstance(p, str):
                result += self.emptyline()
            elif p.language is not None:
                result += self.code(p.indent, p.content, p.language)
            elif p.ordered is not None:
                result += self.ordered(p.indent, p.id, p.ids, p.content) if p.ordered else self.unordered(p.indent, p.content)
            else:
                result += self.plain(p.indent, p.content)
        return result

class RstRenderer(ArticleRenderer):
    def __init__(self, paragraphs):
        super().__init__(paragraphs)

    def ordered(self, indent, id, numids, content):
        return "\n" + super().ordered(indent, id, numids, CITE.sub(r" \1 ", content))

    def unordered(self, indent, content):
        return "\n" + super().unordered(indent, CITE.sub(r" \1 ", content))

    def code(self, indent, content, language):
        lines = CITE.sub(r" \1 ", content).split("\n")
        result = '\n' + ' ' * indent + ".. code-block:: "
        if language is not None:
            result += language
        result += "\n"

        for line in lines:
            result += "\n" + ' ' * 4 + line

        result += "\n\n"
        result += ' ' * indent + '..' + "\n"

        return result

    def plain(self, indent, content):
        return super().plain(indent, CITE.sub(r" \1 ", content))

class Section:
    def __init__(self, key, value, body):
        self.key = key
        self.value = value
        self.body = body

class MultiSection:
    def __init__(self, doc):
        self.sections = {}
        key = None
        val = None
        body = ""
        for line in doc.split("\n"):
            if line.startswith("! "):
                if key is not None:
                    if body == "": body = None
                    self.commit(Section(key, val, body))
                key = None
                val = None
                body = ""
                tks = WS.split(line, 2)
                if len(tks) > 1 and tks[1].strip() != "":
                    key = tks[1].strip()
                if len(tks) > 2 and tks[2].strip() != "":
                    val = tks[2].strip()
            else:
                if body != "": body += "\n"
                body += line
        if key is not None:
            if body == "": body = None
            self.commit(Section(key, val, body))

    def commit(self, section):
        if section.key in self.sections:
            self.sections[section.key].append(section)
        else:
            self.sections[section.key] = [section]

    def __iter__(self):
        return self.sections.__iter__()
    
    def hasAttr(self, name):
        try:
            self.__getattr__(name)
            return True
        except:
            return False

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        if name in self.sections:
            return self.sections[name][0]
        if name.endswith("list") and name[:-4] in self.sections:
            return self.sections[name[:-4]]
        raise KeyError("Attribute `{}` does not exist".format(name))

class DocList:
    def __init__(self, filename = None):
        self.doclist = []
        self.endocdict = {}
        self.chdocdict = {}
        if filename is not None:
            if isinstance(filename, str):
                self.read(filename)
            else:
                for fn in filename:
                    self.read(fn)
        self.doclist.sort()

    def read(self, filename):
        if filename is not None:
            with open(filename, "r", encoding="utf-8") as f:
                entext = ""
                chtext = ""
                name = ""
                for line in f.readlines():
                    if line.startswith("<"):
                        entext += line[1:]
                    elif line.startswith(">"):
                        chtext += line[1:]
                    else:
                        if name != "": self.add(name, entext, chtext)
                        name = line.strip()
                        entext = ""
                        chtext = ""
                self.add(name, entext, chtext)

    def write(self, filename):
        with open(filename, "w") as f:
            for name in self.doclist:
                endoc = self.endocdict.get(name)
                chdoc = self.chdocdict.get(name)
                if endoc is not None or chdoc is not None:
                    f.write(name + "\n")
                    if endoc is not None:
                        for line in endoc.split("\n"):
                            f.write("<" + line + "\n")
                    if chdoc is not None:
                        for line in chdoc.split("\n"):
                            f.write(">" + line + "\n")


    def add(self, name, endoc, chdoc):
        self.doclist.append(name)
        if name != "":
            if endoc != "":
                self.endocdict[name] = endoc
            if chdoc != "":
                self.chdocdict[name] = chdoc

    def endoc(self, name):
        return self.endocdict.get(name)

    def chdoc(self, name):
        return self.chdocdict.get(name)

    def enmultisection(self, name):
        doc = self.endoc(name)
        return None if doc is None else MultiSection(doc)

    def chmultisection(self, name):
        doc = self.chdoc(name)
        return None if doc is None else MultiSection(doc)

    def __iter__(self):
        return self.doclist.__iter__()        
