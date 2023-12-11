# Divine Inference

## Structure

- [.github](.github) - *GitHub Actions*
- [kubernetes](kubernetes) - *Kubernetes manifests*
- [assets](assets) - *Model weights, etc.*
- [utils.py](utils.py) - *Contains the code for the inference and training processes.*
- [environment.yml](environment.yml) - *Conda environment*
- [log.py](log.py) - *Initializes the logger*
- [sp.py](sp.py) - *A configuration file containing a list of equities to be used in the inference process*
- [skaffold.yml](skaffold.yml) - *Skaffold configuration file called by Google Cloud*

**Note:** files prepended with `workflow` are used directly by the Docker image build process.

**Note:** [environment.yml](environment.yml) is used by the Docker image build process 
and can be created using `conda env export > environment.yml`.

**Note:** the [logs](logs) and [notebooks](notebooks) directories remain local and are not pushed to GitHub.


