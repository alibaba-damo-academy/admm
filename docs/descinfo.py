PACKAGE_NAME = "admm"
PACKAGE_VERSION = "1.0.0"
LIBNAME = "admm"
DESCRIPTION = "ADMM - Automatic Decomposition Method by MindOpt"
LICENSE = "MIT"
AUTHOR = "MindOpt Team"
AUTHOR_EMAIL = "solver.damo@list.alibaba-inc.com"
MAINTAINER = "MindOpt Team"
MAINTAINER_EMAIL = "solver.damo@list.alibaba-inc.com"
URL = "https://github.com/alibaba-damo-academy/admm"
KEYWORDS = ["admm", "optimization", "convex", "cython", "python"]
INSTALL_REQUIRES = ["numpy>=1.20.0", "scipy>=1.7.0", "admmlib>=2026.4.4"]
PROJECT_URLS = {
    "Homepage": URL,
    "Documentation": "https://admm.readthedocs.io",
    "Repository": URL,
    "Issues": URL + "/issues",
}
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Mathematics",
]

def generate_definition_hrst():
    import os
    file_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path, "definition.hrst"), "w") as f:
        f.write(f"""
.. |version| replace:: {PACKAGE_VERSION}
.. |version-full| replace:: {PACKAGE_VERSION}
.. |ADMM| replace:: **ADMM**
.. |ADMM-version| replace:: |ADMM| |version|
""")

if __name__ == "__main__":
    generate_definition_hrst()
