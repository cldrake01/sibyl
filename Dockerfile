FROM mambaorg/micromamba:latest
LABEL authors="collin"

ENTRYPOINT ["top", "-b"]
