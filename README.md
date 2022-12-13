# Time Series Classification Based on Fuzzy Cognitive Maps and Multi-class Decomposition with Ensembling

## Requirements
Project usage requires python in version `3.10.0`. In order to run classification tests, all the required libraries need to be installed. Those libraries can be installed using the `requirements.txt` file provided in the project root directory:
```
pip install -r requirements.txt
```

## Usage
### Data Preparation
The project was designed to test classification on the datasets obtained from the https://www.timeseriesclassification.com page. In order to download such a dataset in a format suitable for project usage in the project root directory, run the following:
```
python -m data_converter -v DATASET_NAME
```
where DATASET_NAME is the name of the dataset on https://www.timeseriesclassification.com. Dataset files will be stored in the `data` directory.

In order to test dataset classification, a config file needs to be prepared. This file should be placed in the `configs` directory. Following is an example config file:
```
[BaseClassifier]
MovingWindowSize = 6
MovingWindowStride = 1
FCMConceptCount = 4

[FuzzyCMeans]
M = 2
Error = 1e-8
Maxiter = 1e6

[GeneticAlgorithm]
MaxNumIteration = 800
PopulationSize = 800
MaxIterationWithoutImprov = 50
MutationProbability = 0.05

[GeneticAlgorithmRun]
NoPlot = true
DisablePrinting = true
DisableProgressBar = true
```
The most important parameters are [MovingWindowSize] and [FCMConceptCount]. Others can be left unchanged.

### Dataset Classification
Datasets present in the `data` directory can be classified with the `classifier_pipeline ` tool. All the details about this tool's usage can be obtained with 
```
python -m classifier_pipeline -h 
```
Following is an example of `classifier_pipeline` tool usage (provided that the `data_converter` module was used to download and process the ACSF1 dataset):
```
python -m classifier_pipeline ACSF1 --methods 1_vs_all asymmetric_1_vs_1 symmetric_1_vs_1 combined_symmetric_1_vs_1 --configs 0.ini 1.ini --test-length-fractions 1 0.8 0.6 -v
```

Results of classification are stored in a `.csv` file in the `results` directory. Each row of this file contains information about the configuration of the test case along with classification accuracy on test and train datasets.
