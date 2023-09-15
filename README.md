# Credit Default Prediction Demo


![](raw/latest/images/credit.png?inline=true)

For this demo we'll use the freely available Statlog (German Credit Data) Data Set, which can be downloaded from [Kaggle](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). 
This dataset classifies customers based on a set of attributes into two credit risk groups - good or bad. The majority of the attributes in this data set are categorical, and they are symbolically encoded.

We also use SMOTE to balance and upsample the data, leading to a final training set of approximately 700MB. To succesfully handle this amount of data we leverage two Domino features:

* [Domino Datasets](https://docs.dominodatalab.com/en/5.0/user_guide/0a8d11/domino-datasets/) --- high-performance, versioned, and structured filesystem storage in Domino, which can also make data available to multiple backend compute instances for the purposes of distributed processing.

* [Ray](https://docs.ray.io/en/latest/) --- Ray is a unified framework for scaling AI and Python applications. In this project we use it to distribute the model fitting process and speed up the hyperparameter optimisation.

* [Distributed XGBoost on Ray](https://github.com/ray-project/xgboost_ray) --- XGBoost-Ray is a distributed backend for XGBoost, built on top of distributed computing framework Ray. XGBoost-Ray

    * enables multi-node and multi-GPU training
    * integrates seamlessly with distributed hyperparameter optimization library Ray Tune
    * comes with advanced fault tolerance handling mechanisms, and
    * supports distributed dataframes and distributed data loading
    
* [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) --- Tune is a Python library for experiment execution and hyperparameter tuning at scale. We use it for performing hyperparameter tuning of the XGBoost model.

* [ELI5](https://eli5.readthedocs.io/en/latest/overview.html) --- A Python framework, which helps to debug machine learning classifiers and explain their predictions.

## Model training

This project contains the following assets relating to model training:

* [data_generation.ipynb](view/data_generation.ipynb) --- data generation notebook. This notebook upsamples and prepares the model training and testing datasets. There is no need to execute it as everything is already in place. It is included in the project for completeness.

* [training.ipynb](view/training.ipynb) --- training notebook. This notebook trains the predictive model and also shows basic model explainability. To run this notebook you'll need to configure a [Domino Workspace](https://docs.dominodatalab.com/en/latest/user_guide/867b72/workspaces/). 

    * Note, that the workspace needs to have an on-demand Ray cluster attached. When attaching the cluster make sure you specify 2 worker nodes. Also, select Medium (4 cores 15 GiB RAM) as the hardware tier for botht the head and the worker nodes. You can select a different number of nodes and different hardware tiers, but don't forget to change the `RAY_ACTORS` and `RAY_CPUS_PER_ACTOR` variables in the Python code to fully utilize the new cluster configuration.
    * Click this button to create a workspace [![Run Notebook](raw/latest/images/create_workspace.png)](/workspace/:ownerName/:projectName?showWorkspaceLauncher=True)

* [train_model.py](view/train_model.py) --- model training script. This is a Python that contains all the code from [training.ipynb](view/training.ipynb), but in a form that makes it easier for executing as a [Domino Job](https://docs.dominodatalab.com/en/latest/user_guide/942549/jobs/). The script also accepts the following command line arguments:

    * `ray_actors` --- number of Ray actors (default value is 3)
    * `cpus_per_actor` --- number of CPUs per actor (4 by default)
    * `tune_samples` --- how many models to train over the hyperparameter space (10 by default)
    
    
    Note, that for this script to work properly it would also need an attached Ray cluster (same configuration as for the training notebook)


## Model API

After the model is trained, it can be deployed as a [Model API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/model-apis/). The following file contains the scoring function.

* [score.py](view/score.py) --- scoring function for the Model API. The name of the function is `predict_credit`, and for simplicity it only accepts the 10 most significant features (based on the explainability analysis).

Here is a sample JSON paylod, which can be used to test the Model API:

```
    {
      "data": {
        "checking_account_A14": 0,
        "credit_history_A34": 0,
        "property_A121": 0,
        "checking_account_A13": 0,
        "other_installments_A143": 1,
        "debtors_guarantors_A103": 0,
        "savings_A65": 0,
        "age": 0.285714,
        "employment_since_A73": 1,
        "savings_A61": 1
      }
    }
```

## Setup instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present:

### Ray Workspace 2.4.0

**Environment Base** 

`quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.3-ray2.4.0-domino5.7`

**Dockerfile Instructions**

```
USER ubuntu

RUN pip install xgboost-ray==0.1.15 \
    && pip install eli5==0.13.0
    
RUN pip install altair==5.1.1 \
    && pip install --user streamlit==1.26.0 argparse==1.4.0 imblearn==0.0
```

**Pluggable Workspace Tools**


```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
  supportedFileExtensions: [ ".ipynb" ]
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  /opt/domino/workspaces/jupyterlab/start ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```

### Ray Cluster 2.4.0

**Environment Base** 

`rayproject/ray-ml:2.4.0-py39-gpu`

**Supported Cluster Settings**

Ray


**Dockerfile Instructions**
```
USER root
RUN usermod -u 12574 ray 

RUN pip install --upgrade xgboost==1.6.2
```
