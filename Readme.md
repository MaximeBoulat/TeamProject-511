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

## Stages

### Stage 1: Data acquisition

- **Input**: unzipped files at `build/datasets/dataset_4composers`
- **Process**: traverse and build an index
- **Output**: a df with 2 columns: "composer" and "path"
- **Components**: `DataBuilder`

### Stage 2: Data preprocessing

- **Input**: the df from Stage 1
- **Process**: remove duplicates and unuseable files
- **Output**: a cleaned df
- **Components**: `DataPreprocessor`

### Stage 3: Data splitting

- **Input**: the df from Stage 2
- **Process**: Split twice using the `TEST_SIZE` and `VAL_SIZE` parameters
- **Output**: 3 pairs: f_train, y_train, f_val, y_val, f_test, y_test
- **Components**: `DataSplitter`

### Stage 4: Feature extraction

- **Input**: the f_train, y_train, f_val, y_val, f_test, y_test pairs from Stage 3
- **Process**: split into windows and convert windows into the designated data type for each model type (piano roll for cnn and sequence for lstm)
- **Output**: 3 data structures (`train_array`, `val_array` and `test_array`) which contain for cnn and lstm:
  - `x`, `y` and `fid`
- **Components**: `FeatureExtractor`

### Stage 5: Data augmentation

- **Input**: The train arrays from Stage 4
- **Process**: build pipelines for each array, for each model type
- **Output**: 6 pipelines
- **Components**: `PipelineBuilder`

### Stage 6: Model training

- **Input**: the val_array from Stage 4, class weights and the pipelines from Stage 5
- **Process**: First check if the training has already run by checking in the `build/artifacts` directory. If not, train the models and save the models in the `build/artifacts` directory. If yes, load the models from the `build/artifacts` directory, then put the training graph pngs in `build/visualizations/training`
- **Output**: artifacts (models + history jsons)
- **Components**: `ModelTrainer`

### Stage 7: Model evaluation

- **Input**: the test_array from Stage 4, the models from Stage 6
- **Process**: evaluate the models on the test set and put the evaluation graph pngs in `build/visualizations/evaluation`. Also do the side-by-side comparison of the models and put the comparison graph pngs in `build/visualizations/evaluation`.
- **Output**: evaluation results
- **Components**: `ModelEvaluator`

