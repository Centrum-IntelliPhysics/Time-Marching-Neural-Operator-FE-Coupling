# Table of Content 
[General information](#-General-information)


# General information


# DeepONet_FEM_Coupling
Attached code in relevant paper: [Accelerating Multiscale Modeling with Hybrid Solvers: Coupling FEM and Neural Operators with Domain Decomposition](
https://doi.org/10.48550/arXiv.2504.11383)

## Virtual Environment   
Install the JAX on website: https://docs.jax.dev/en/latest/installation.html  
CUDA version:
<pre><code>pip install --upgrade pip
# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
</code></pre>  


Install FEniCSx on website: https://fenicsproject.org/download/  
Conda install:
<pre><code>conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
</code></pre>


