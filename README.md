### Abstract
pyVISCOUS is the open-source code of VISCOUS in Python. VISCOUS (VarIance-based Sensitivity analysis using COpUlaS) is a variance-based global sensitivity analysis framework. It was developed by Sheikholeslami et al. ([2021](https://doi.org/10.1029/2020WR028435)) and improved by Liu et al. (2023).The advantage of VISCOUS is that it can use existing model input-output data (e.g., water model parameters-responses) to estimate the first- and total-order Sobol’ sensitivity indices. 

VISCOUS is a given-data approach. It can be implemented regardless of whether the underlying relationship or mechanism between input and output data is known. It also enables the application of gloabl sensitivity analysis to computationally intensive models by generating reliable sensitivity estimates with minimal computational resources. 

Within the VISCOUS framework, the following steps are included. Details can be found in Liu et al. (2023).

![flowchart](https://github.com/CH-Earth/pyviscous/assets/48458815/2e8f7575-41d4-4e6a-bac8-fadc2a5b9c7a)

### Installation
#### From PyPI
```pip install pyviscous```

#### From source

Clone pyviscous with: ```git clone https://github.com/CH-Earth/pyviscous.git```

Then navigate to the pyviscous directory and install with: ```python setup.py install```

### Examples
We provide four example notebooks in the example directory. In each example, there are scripts to generate input-output data, set up and run VISCOUS, and evaluate the sensitivity results.

### References
Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Knoben, W. J. M., Freer, J., Marsh C. B., & Papalexiou, S. M. (2023) pyVISCOUS: An open-source tool for computationally frugal global sensitivity analysis. (Submitted to Water Resources Research)

Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. Water Resources Research, 57, e2020WR028435, https://doi.org/10.1029/2020WR028435

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
