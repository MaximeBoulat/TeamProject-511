# Introduction

This repo contains the source code and the technical report for USD's AAI-511 group project for Team . 

The repo is organized as follows:

```
/
└── Final.ipynb // The source code as a Jupyter notebook
└── src/  // The same source code as a Python app, broken down in stages/classes
└── Paper/  // The source files of the technical report and the final rendered version
└── <teammate name>/  // Private teammate folders
└── build/   // build artifacts + designated dataset drop point
```

# Usage

The full pipeline can be executed both as a notebook (`Final.ipynb`) and as a regular python app (`python3 src/main.py` from the root folder). 

Both approaches require that a build folder be created and that the unzipped dataset (which can be found in Robert's folder) be placed in it as follows:

```
/
└── build/
    └── datasets/
            └── dataset_4composers/
                ├── Bach/
                ├── Beethoven/
                ├── Chopin/
                └── Mozart/
```

# Architecture



