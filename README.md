### Abstract
pyVISCOUS is the open-source code of VISCOUS in Python. VISCOUS is a computationally frugal variance-based global sensitivity analysis framework ([Sheikholeslami et al., 2021](https://doi.org/10.1029/2020WR028435)). VISCOUS consists of two elements: developing a probability model to describe the relationship between model inputs (e.g., model parameters) and outputs (e.g., model responses); and computing the Sobol sensitivity indices based on the developed probability model.

### Installation
#### From PyPI
```pip install pyviscous```

#### From source

Clone pyviscous with: ```git clone https://github.com/h294liu/pyviscous.git```

Then navigate to the pyviscous directory and install with: ```python setup.py install```

### Examples
We provide four example notebooks in the example directory. In each example, there are scripts to generate input-output data, set up and run VISCOUS, and evaluate the sensitivity results.

### References
Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. Water Resources Research, 57, e2020WR028435, https://doi.org/10.1029/2020WR028435

Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Freer, J., Knoben, W. J. M., & Papalexiou, S. M. (2022) pyVISCOUS: An open-source tool for computationally frugal global sensitivity analysis. (Submitted to Water Resources Research)

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
