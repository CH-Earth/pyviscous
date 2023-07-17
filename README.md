### Abstract
pyVISCOUS is the open-source code of VISCOUSm in Python. VISCOUSm is an improved copula-based framework for efficient global sensitivity analysis. It has the advantage that it can use existing model input (e.g., model parameters) and output (e.g., model responses) data to estimate Sobol’ sensitivity indices. In comparison with the VISCOUS of Sheikholeslami et al. ([2021](https://doi.org/10.1029/2020WR028435)), VISCOUSm improves the handling of marginal densities of the GMCM (Gaussian mixture copula model). 

Within the VISCOUSm framework, the following steps are included.

![flowchart](https://github.com/CH-Earth/pyviscous/assets/48458815/2e8f7575-41d4-4e6a-bac8-fadc2a5b9c7a)

### Installation
#### From PyPI
```pip install pyviscous```

#### From source

Clone pyviscous with: ```git clone https://github.com/CH-Earth/pyviscous.git```

Then navigate to the pyviscous directory and install with: ```python setup.py install```

### Examples
We provide four example notebooks in the example directory. In each example, there are scripts to generate input-output data, set up and run VISCOUSm, and evaluate the sensitivity results.

### References
Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Freer, J., Knoben, W. J. M., Marsh C. B., & Papalexiou, S. M. (2022) pyVISCOUS: An open-source tool for computationally frugal global sensitivity analysis. (Submitted to Water Resources Research)

Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. Water Resources Research, 57, e2020WR028435, https://doi.org/10.1029/2020WR028435

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
