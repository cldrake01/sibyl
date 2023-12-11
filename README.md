# Divine Inference

## Structure

- [.github](.github) - *GitHub Actions*
- [assets](assets) - *Model weights, etc.*
- [kubernetes](kubernetes) - *Kubernetes manifests*
- [environment.yml](environment.yml) - *Conda environment*
- [skaffold.yml](skaffold.yml) - *Skaffold configuration file called by Google Cloud*
- [utils.py](utils.py) - *Contains the code for the inference and training processes.*
- [log.py](log.py) - *Initializes the logger*
- [sp.py](sp.py) - *A configuration file containing a list of equities to be used in the inference process*

**Note:** files prepended with `workflow` are used directly by the Docker image build process.

**Note:** [environment.yml](environment.yml) is used by the Docker image build process 
and can be created using `conda env export > environment.yml`.

**Note:** the [logs](logs) and [notebooks](notebooks) directories remain local and are not pushed to GitHub.


