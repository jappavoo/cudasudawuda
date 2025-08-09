# CSW Open Data Hub Notebook based images

How these images are accessed on the NERC MOC OpenShift
test and academic clusters

1. A selectable notebook was created using the  image stream yaml: yaml/csw-dev-F25.yaml
2. This notebook points the http://quay.io/jappavoo/csw-dev:nerctest-current

## Image Construction and Lineage

The image is created using podman and the `Containerfile` in this directory

The following is the current lineage:
1. 
- ./Containerfile
- IMAGE: quay.io/nerc-images/cuda-jupyter-tensorflow:latest
- SRC: https://github.com/nerc-images/cuda-jupyter-tensorflow

2. ODH Image (Open Data Hub)
-  https://github.com/nerc-images/cuda-jupyter-tensorflow/blob/main/Containerfile
- IMAGE: quay.io/modh/cuda-notebooks:cuda-jupyter-tensorflow-ubi9-python-3.11-20250806
- SRC: https://github.com/opendatahub-io/notebooks/blob/main/jupyter/tensorflow/ubi9-python-3.11/Dockerfile.cuda
- DOC: https://github.com/opendatahub-io/notebooks/wiki/Workbenches

This image uses several bases to build its self we will focus on the lineage of CUDA it uses so that we can determine 
how to add consitent CUDA tools
- FROM registry.access.redhat.com/ubi9/go-toolset:latest AS mongocli-builder
- SRC https://catalog.redhat.com/software/containers/ubi9/go-toolset/61e5c00b4ec9945c18787690?container-tabs=dockerfile
- FROM registry.access.redhat.com/ubi9/python-311:latest AS base
- SRC https://catalog.redhat.com/software/containers/ubi9/python-311/63f764b03f0b02a2e2d63fff?container-tabs=dockerfile

The CUDA components are added by a carefull copy and paste of 
the dockerfiles provided by NVIDIA (https://gitlab.com/nvidia/container-images/cuda/-/tree/master/doc#overview-of-images)
These can be found here:
base:
- https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubi9/base/Dockerfile
runtime:
- https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubi9/runtime/Dockerfile
devel:
- https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubi9/devel/Dockerfile
cudnn:
- https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubi9/devel/cudnn/Dockerfile
