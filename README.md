### Abstract
pyVISCOUS is the open-source code of VISCOUS in Python. VISCOUS (VarIance-based Sensitivity analysis using COpUlaS) is a variance-based global sensitivity analysis framework. It was developed by Sheikholeslami et al. ([2021](https://doi.org/10.1029/2020WR028435)) and improved by Liu et al. (2023).


As a “given-data” method, VISCOUS uses existing model input and output data (e.g., model parameters and responses) to provide useful approximations of the first- and total-order Sobol’ sensitivity indices. The greatest advantage of VISCOUS over other given-data global sensitivity analysis methods is that VISCOUS does not require the input data follow any specific sampling strategies. The input-output data can be from the previous model runs generated from other modeling purposes, such as uncertainty propagation and model calibration.


Within the VISCOUS framework, the following steps are included. Details can be found in Liu et al. (2023).

![flowchart](https://github.com/CH-Earth/pyviscous/assets/48458815/2e8f7575-41d4-4e6a-bac8-fadc2a5b9c7a)

### Install pyviscous
**From PyPI**: ```pip install pyviscous```

**From source**:

Clone pyviscous with: ```git clone https://github.com/CH-Earth/pyviscous.git```

Then navigate to the pyviscous directory and install with: ```python setup.py install```

### Do not want to install pyviscous
If you do not want to install pyviscous, you can still use it by adding the pyviscous source code to the system path. For example,

```
import sys
sys.path.insert(<path_to_directory>)
import pyviscous
```

<path_to_directory> is the path to the folder where the pyviscous repository is located on your computer. The first two lines add the path of the repository directory to the system path so that Python can also look for the package in that directory if it doesn’t find it in its current directory. 

**Important note**: To use pyviscous in this way, please make sure that you have installed all the required Python packages listed in setup.py file (i.e., *numpy*, *pandas*, *scipy*, *scikit-learn*, *copulae*, *matplotlib*, *jupyter*). Please install copulae via pip, not conda. This is because the conda distribution of copulae does not properly include its full source code/functions. We will remind the developer of copulae to fix this. 

### Examples
We provide four example notebooks in the example directory. In each example, there are scripts to generate input-output data, set up and run VISCOUS, and evaluate the sensitivity results.

### How to cite pyVISCOUS code
Hongli Liu, Martyn P. Clark, Shervan Gharari, Razi Sheikholeslami, Jim Freer, Wouter J. M. Knoben, Christopher B. Marsh, & Simon Michael Papalexiou. (2023). pyVISCOUS. Zenodo. https://doi.org/10.5281/zenodo.8179325

### References
Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Freer, J., Knoben, W. J. M., Marsh C. B., & Papalexiou, S. M. (2023) An improved copula-based framework for efficient global sensitivity analysis. (Submitted to *Water Resources Research*)

Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. *Water Resources Research*, 57, e2020WR028435, https://doi.org/10.1029/2020WR028435

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
