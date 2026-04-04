"""
MindOpt Document Editor - English Only Version

A web-based editor for maintaining doc.data (API documentation source).
Simplified for open-source release - English only, no prefix markers.

Usage:
    python edit.py
    
Access:
    http://localhost:8080
"""
import web
import json
from markdown import *

urls = (
    '/edit.js', 'js',
    '/save', 'save',
    '/preview', 'preview',
    '/', 'edit'
)
app = web.application(urls, globals())

SCRIPT = r'''
function show(e, lines) {
    value = "";
    for (var i in lines) {
        if (i > 0) value += "\n";
        value += lines[i];
    }
    e.value = value;
}

window.onload = function() {
    list_ids();

    document.getElementById("en").addEventListener("input", function(e) {
        if (window["cdid"] == "") {
            e.target.value = "";
            return;
        }
        data[cdid]["en"] = e.target.value;
    });

    document.getElementById("new").addEventListener("click", function(e) {
        var id = prompt("Input doc id").trim();
        if (id.split(/\s+/).length > 1) {
            window.alert("id must not contain spaces");
        } else if (window.data.hasOwnProperty(id)) {
            window.alert("id already exists");
        } else {
            window.data[id] = {};
            list_ids();
            openid(id);
        }
    });

    document.getElementById("delete").addEventListener("click", function(e) {
        if (window.confirm("Are your sure?")) {
            delete window["data"][window["cdid"]];
            cdid = "";
            list_ids();
        }
    });

    document.getElementById("prev").addEventListener("click", function(e) {
        var value = document.getElementById("en").value.trim();
        if (value == "") {
            window.alert("no en content found");
            return;
        }
        render(value);
    });

    document.getElementById("save").addEventListener('click', function(e) {
        save();
    });

    document.addEventListener('keydown', function(e) {
        if (e.key == "s" && e.metaKey)
        {
            e.preventDefault();
            save();
        }
    });
}
'''

HTML = r'''
<html>
<head>
<title>MindOpt Document Editor</title>
<script>
    var data = __saved_data__;
    var cdid = "";
</script>
<script src="/edit.js"></script>
<style>
    #main {
        width: 100%;
        height: 100%;
    }

    .navi {
        width: 180px;
        height: 100%;
        position: fixed;
        top: 0px;
        left: 0px;
        border-right: solid 1px black;
        display: inline-block;
    }

    .en {
        width: calc(50% - 120px);
        height: 100%;
        position: fixed;
        top: 0px;
        left: 180px;
        border-right: dashed 1px gray;
        display: inline-block;
        overflow-y: scroll;
    }

    .preview {
        width: calc(50% - 60px);
        height: 100%;
        position: fixed;
        top: 0px;
        left: calc(50% + 60px);
        padding-left: 5px;
        display: inline-block;
        overflow-y: scroll;
    }

    .navi > :first-child {
        text-align: center;
        font-weight: bold;
        padding: 10px;
        border-bottom: solid 1px black;
    }

    .navi > :nth-child(2) {
        padding: 5px;
    }

    .navi > :nth-child(2) > * {
        width: 100%;
    }

    .navi > :nth-child(3) {
        overflow-y: scroll;
        height: calc(100% - 100px);
    }

    .did {
        padding: 5px;
        cursor: pointer;
    }

    .did:hover {
        background-color: #eee;
    }

    .did.active {
        background-color: #ccc;
    }

    .en > :first-child {
        font-weight: bold;
        padding: 10px;
        border-bottom: solid 1px black;
    }

    .en textarea {
        width: 100%;
        height: calc(100% - 50px);
        border: none;
        resize: none;
        padding: 10px;
        font-family: monospace;
        font-size: 14px;
    }

    .preview > :first-child {
        font-weight: bold;
        padding: 10px;
        border-bottom: solid 1px black;
    }

    .preview > :nth-child(2) {
        padding: 10px;
    }
</style>
</head>
<body>
<div id="main">
    <div class="navi">
        <div>Document Editor</div>
        <div>
            <button id="new">New</button>
            <button id="delete">Delete</button>
        </div>
        <div id="ids">
        </div>
    </div>
    <div class="en">
        <div>English Documentation</div>
        <textarea id="en"></textarea>
    </div>
    <div class="preview">
        <div>Preview</div>
        <div id="preview"></div>
    </div>
</div>
</body>
</html>
'''

class js:
    def GET(self):
        web.header('Content-Type', 'application/javascript')
        return SCRIPT

class edit:
    def GET(self):
        web.header('Content-Type', 'text/html')
        data = {}
        try:
            with open("doc.data", "r") as f:
                cur_id = ""
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    # All content is English, no prefix needed
                    if cur_id != "":
                        data[cur_id]["en"] = data[cur_id].get("en", "") + line
                    else:
                        cur_id = line.strip()
                        data[cur_id] = {}
        except FileNotFoundError:
            pass
        
        return HTML.replace("__saved_data__", json.dumps(data, indent=2))

class save:
    def POST(self):
        data = web.data()
        with open("doc.data", "w") as f:
            f.write(data.decode('utf-8'))
        return "OK"

class preview:
    def POST(self):
        data = web.data().decode('utf-8')
        lines = data.split("\n")
        result = ""
        for line in lines:
            if line.strip() == "":
                continue
            # All content is English, no prefix needed
            result += line + "\n"
        try:
            doc = parseArticle(result)
            result = RstRenderer(doc).render()
        except Exception as e:
            result = str(e)
        return result

if __name__ == "__main__":
    app.run()
