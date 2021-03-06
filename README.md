# mlflow-project-template
mlflow-project-template

# MLflow-project-template
MLflow project template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

### To access conda from c drive to d, through bash
---------------------
    . C:/Users/DELL/Anaconda3/etc/profile.d/conda.sh
---------------------

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05 - Create conda.yaml file -
```bash
conda env export > conda.yaml
```
### Copy a template of python files
'''bash
cp src/stage_00_template.py src/stage_01_get_data.py
''' 
### STEP 06- commit and push the changes to the remote repository

## MLFlow commands

### Command to run MLFlow Project files
'''bash
mlflow run . --no-conda
''' 

### runany specific entry point in MLFlow Project files
'''bash
mlflow run . -e get_data --no-conda
''' 

### runany specific entry point with certain configuration in MLFlow Project files
'''bash
mlflow run . -e get_data -P configs/your_config.yaml --no-conda
''' 

