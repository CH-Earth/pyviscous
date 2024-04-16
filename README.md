### PyVISCOUS: Open-Source VISCOUS Code in Python
VISCOUS is a variance-based global sensitivity analysis framework (Sheikholeslami et al. 2021, Liu et al. 2024). Functioning as a "given-data" method, VISCOUS utilizes existing model input and output data, such as water model parameters and responses, to provide valuable approximations of first- and total-order Sobol’ sensitivity indices. The input-output data can be from previous model runs conducted for various modeling purposes (e.g., model calibration and uncertainty analysis). 

### Install pyviscous
**From PyPI**:
```
pip install pyviscous
```

**From source**:
```
# Fork the repository on GitHub (assuming you have the rights to do so).
# You can fork it manually from the GitHub interface.

# Clone your forked repository
git clone https://github.com/your_username/pyviscous.git

# Navigate to the pyviscous repository directory
cd pyviscous

# Install
python setup.py install
```
Replace *your_username* with your GitHub username. This assumes that you have forked the repository on GitHub manually.

**Do not want to install pyviscous?**<br>
If you prefer not to install pyviscous, you can still use it by adding the pyviscous source code to the system path. For example:
```
import sys
sys.path.insert(path_to_directory)
import pyviscous
```
Replace *path_to_directory* with the absolute path to the folder where the pyviscous repository is located on your computer. **Note:** before utilizing pyviscous in this manner, ensure that you have installed all the necessary Python packages as listed in the setup.py file. These packages include **numpy**, **pandas**, **scipy**, **scikit-learn**, **copulae**, **matplotlib**, **jupyter**. 

**NOTE**: Please install the **copulae** package using **pip**, not **conda**. The conda distribution of **copulae** lacks its full source code/functions. We recommend using pip for a comprehensive installation. We will notify the copulae developer about this issue for resolution. 

### Examples
We provide five demonstration notebooks in the example directory, including the Rosenbrock function and four Sobol’ functions from Liu et al. (2024). Additionally, a real case study of the **Bow at Banff basin, Alberta, Canada**, is included to show the real-world application of VISCOUS. Each example includes scripts for input-output data generation or reading, VISCOUS setup and execution, and evaluation of sensitivity results.

### Credits
VISCOUS was originally developed by Sheikholeslami et al. (2021) and enhanced by Liu et al. (2023). 
- Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. *Water Resources Research*, 57, e2020WR028435. https://doi.org/10.1029/2020WR028435
- Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Freer, J., Knoben, W. J. M., Marsh C. B., & Papalexiou, S. M. (2024) An improved copula-based framework for efficient global sensitivity analysis. *Water Resources Research*, 60, e2022WR033808. https://doi.org/10.1029/2022WR033808

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
