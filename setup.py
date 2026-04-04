"""ADMM Python package setup script."""
import os
import platform
import sys

from setuptools import setup, Extension
from Cython.Build import cythonize
from stubgen_pyx import StubgenPyx
from pathlib import Path

root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
os.chdir(root_path)

try:
    from docs import dynamic
    from docs.descinfo import PACKAGE_NAME, PACKAGE_VERSION, DESCRIPTION, LICENSE, LIBNAME
    from docs.descinfo import AUTHOR, AUTHOR_EMAIL, MAINTAINER, MAINTAINER_EMAIL
    from docs.descinfo import URL, PROJECT_URLS, CLASSIFIERS, KEYWORDS, INSTALL_REQUIRES

    import admmlib

    dep_lib_path = str(admmlib.lib_dir)
    include_dir = str(admmlib.include_dir)

    saved_cwd = os.getcwd()
    try:
        os.chdir(dep_lib_path)
        dynamic.genPxiAndDoc(os.path.join(dep_lib_path))
    finally:
        os.chdir(saved_cwd)
    dynamic.mergeAndFillDoc()
except ImportError as e:
    raise RuntimeError("Install admmlib with pip first: {}".format(e))

compile_options = []
link_options = []

if platform.system() == "Linux":
    link_options = [
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath,$ORIGIN/../lib",
        "-Wl,-rpath,$ORIGIN/../admmlib/lib",
    ]
elif platform.system() == "Darwin":
    compile_options += [
        "-Wno-unreachable-code",
        "-Wno-unreachable-code-fallthrough",
    ]
    link_options = [
        "-Wl,-rpath,@loader_path",
        "-Wl,-rpath,@loader_path/../lib",
        "-Wl,-rpath,@loader_path/../admmlib/lib",
    ]
elif platform.system() == "Windows":
    compile_options += ["/std:c17"]

include_dirs = ["src/admm"]
if include_dir:
    include_dirs.append(include_dir)

library_dirs = []
if dep_lib_path:
    library_dirs.append(dep_lib_path)

extensions = [
    Extension(
        "admm.admm",
        sources=["src/admm/mdcp.pyx"],
        include_dirs=include_dirs,
        libraries=[LIBNAME],
        library_dirs=library_dirs,
        extra_compile_args=compile_options,
        extra_link_args=link_options,
        language="c",
    )
]

ext_modules = cythonize(
    extensions,
    compiler_directives={"language_level": "3"},
    include_path=["src/admm"],
)

stubgen = StubgenPyx()

# Generate stub file
pyx_code = Path("src/admm/mdcp.pyx").read_text()
pyi_stub = stubgen.convert_str(pyx_code)
with open("src/admm/admm.pyi", "w") as f:
    f.write(pyi_stub)

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description=DESCRIPTION,
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    project_urls=PROJECT_URLS,
    packages=[PACKAGE_NAME, "udf"],
    package_dir={PACKAGE_NAME: "src/admm", "udf": "udf"},
    package_data={
        PACKAGE_NAME: ["*.so", "*.dylib", "*.dll", "*.pyi"]
    },
    exclude_package_data={
        PACKAGE_NAME: ["*.h", "*.c", "*.cpp"]
    },
    ext_modules=ext_modules,
    zip_safe=False,
)
