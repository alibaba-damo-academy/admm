window.DOC_ROOT = "";
window.VER_ROOT = "";

(function (cb) {
    if (document.readyState!='loading') cb();
    else if (document.addEventListener) document.addEventListener('DOMContentLoaded', cb);
    else document.attachEvent('onreadystatechange', function() {
        if (document.readyState=='complete') cb();
    });
})(function() {
    var url = window.location.pathname + window.location.search;
    var workpath = window.location.pathname.split("/");
    var rootpath = (window.DOCUMENTATION_OPTIONS.URL_ROOT + "../../../..").split("/");
    
    var parts = workpath.length;
    for (var i = 0; i < rootpath.length; i++)
        if (rootpath[i] == "..") parts--;

    window.DOC_ROOT = workpath.slice(0, parts).join("/");
    window.VER_ROOT = workpath.slice(0, parts + 1).join("/");

    var navdiv = document.getElementsByClassName("wy-side-nav-search")[0];

    var lang = document.createElement("div");
    lang.style.fontSize = "0.6em";
    lang.style.display = "inline";

    var buttons = document.createElement("span");
    buttons.style.border = "solid 1px rgb(30, 135, 147)";
    buttons.style.borderRadius = "3px";

    var ch = document.createElement("span");
    ch.innerText = "中文";
    ch.style.backgroundColor = "rgb(30, 135, 147)";
    ch.style.color = "#fff";

    var en = document.createElement("span");
    en.innerText = "EN";
    en.style.paddingLeft = "0.3em";
    en.style.paddingRight = "0.3em";
    en.style.backgroundColor = "rgb(30, 135, 147)";
    en.style.color = "#fff";

    var relpath = url.substring(window.DOC_ROOT.length).toLowerCase();
    enocc = relpath.indexOf("/en");
    cnocc = relpath.indexOf("/cn");
    relpath = url.substring(window.DOC_ROOT.length);

    if (enocc >= 0 || cnocc >= 0)
    {
        var isEn = true;
        if (enocc >= 0 && cnocc >= 0) {
            isEn = enocc <= cnocc;
        } else if (cnocc >= 0) {
            isEn = false;
        }

        var clickable = isEn ? ch : en;

        clickable.style.backgroundColor = "#fff";
        clickable.style.color = "#000";
        clickable.style.cursor = "pointer";

        buttons.appendChild(ch);
        buttons.appendChild(en);
        lang.appendChild(buttons);
        navdiv.children[0].after(lang);

        clickable.addEventListener("click", function(e) {
            occ = isEn ? enocc : cnocc;
            var target = relpath.substring(0, occ);
            var targetLang = isEn ? "/cn" : "/en";
            target += targetLang;
            target += relpath.substring(occ + 3);
            window.location.replace(window.DOC_ROOT + target);
        });
    }

    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var versions = JSON.parse(xhttp.responseText);
            var versiondiv = navdiv.children[navdiv.children.length - 2];
            versiondiv.innerText = "";

            var dropdown = document.createElement("select");

            for (var i in versions) {
                var opt = document.createElement("option");
                opt.innerText = versions[i];
                if (versions[i] == window.DOCUMENTATION_OPTIONS.VERSION)
                    opt.selected = "selected";
                dropdown.appendChild(opt);
            }

            dropdown.style.color = "inherit";
            dropdown.style.fontSize = 'inherit';
            dropdown.style.border = '0';
            dropdown.style.boxShadow = 'none';

            versiondiv.appendChild(dropdown);

            dropdown.addEventListener("change", function(e) {
                var ver = this.value;
                window.location.replace(window.DOC_ROOT + "/" + ver + url.substring(window.VER_ROOT.length));
            });
        }
    };

    xhttp.open("GET", window.DOC_ROOT + "/versions.json", true);
    xhttp.send();
})
