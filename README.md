# Sibyl

## Google Cloud Deployments

We deploy a Kubernetes instance alongside a single Docker container for model training.

- **[divine-inference](https://hub.docker.com/repository/docker/collindrake/divine-inference/general)** -
  *A Kubernetes instance containing [divine-inference](https://hub.docker.com/repository/docker/collindrake/divine-inference/general)
  pods which run the inference process*
- **[divine-erudition]()** -
  *A Docker container that trains the Informer model and distributes the weights to Google Cloud Storage*

## Structure

- ***[.github](.github)*** - *GitHub Actions*
- ***[assets](assets)*** - *Model weights, etc.*
- ***[kubernetes](kubernetes)*** - *Kubernetes manifests*
- [environment.yml](environment.yml) - *Conda environment*
- [skaffold.yml](skaffold.yml) - *Skaffold configuration file called by Google Cloud*
- ***[sibyl/tests](sibyl/tests)*** - *Unit tests*
- ***[sibyl/utils](sibyl/utils)*** - *Utility functions*
- | [preprocessing.py](sibyl/utils/preprocessing.py) - *Preprocessing functions*
- | [retrieval.py](sibyl/utils/retrieval.py) - *Retrieval functions*
- | [tickers.py](sibyl/utils/tickers.py) - *A list of tickers*
- | [weights.py](sibyl/utils/weights.py) - *Retrieves model weights from Google Cloud Storage*
- ***[sibyl/workflows](sibyl/workflows)*** - *Training and inference workflows*
- | [inference.py](sibyl/workflows/inference.py) - *Inference workflow*
- | [training.py](sibyl/workflows/training.py) - *Training workflow*

**Note:** files prepended with `workflow` are used directly by the Docker image build process.

**Note:** [environment.yml](environment.yml) is used by the Docker image build process
and can be created using `conda env export > environment.yml`.

**Note:** the [logs](logs) and [notebooks](notebooks) directories remain local and are not pushed to GitHub.
