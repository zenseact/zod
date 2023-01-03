FROM nvcr.io/nvidia/pytorch:22.12-py3

# Install using poetry
RUN pip install tqdm \
    numpy \
    scipy \
    h5py \
    pyquaternion \
    numpy-quaternion \
    importlib-metadata \
    typer \
    dropbox \
    opencv \
    pyproj \
    matplotlib \
    plotly \
    ipykernel \
    pandas \
    dataclass_wizard