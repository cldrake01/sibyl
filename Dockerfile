FROM mambaorg/micromamba:latest
LABEL authors="collin"

ENTRYPOINT ["top", "-b"]

# Copy the environment.yml file into the container
COPY environment.yml /tmp/environment.yml

# Create the environment using the environment.yml file
RUN conda env create -f /tmp/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# The code to run when container is started:
COPY . /app
WORKDIR /app

# Run your application
CMD ["conda", "run", "-n", "myenv", "python", "workflow.train.py"]

#CMD ["conda", "run", "-n", "myenv", "python", "workflow.inference.py"]