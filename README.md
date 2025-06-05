### PyVISCOUS: Open-Source VISCOUS Code in Python
VISCOUS is a variance-based global sensitivity analysis framework (Sheikholeslami et al. 2021, Liu et al. 2024). Functioning as a "given-data" method, VISCOUS utilizes existing model input and output data, such as water model parameters and responses, to provide valuable approximations of first- and total-order Sobol’ sensitivity indices. The input-output data can be from previous model runs conducted for various modeling purposes (e.g., model calibration and uncertainty analysis). 

### Install pyviscous
**Approach 1: From PyPI**
```
pip install pyviscous
```

**Approach 2: From source**
```
# Clone the repository (replace <your_username> with your GitHub username)
git clone https://github.com/<your_username>/pyviscous.git

# Navigate into the cloned directory
cd pyviscous

# Install
python setup.py install
```
**Note**: This assumes you have forked the repository from GitHub manually.

**Approach 3: Use pyviscous without installing** <br>
If you prefer not to install `pyviscous`, or you'd like to develop or quickly test `pyviscous`, you can directly run it by adding the source directory to your system path:
```
import sys
sys.path.insert(0, "<path_to_directory>")
import pyviscous
```
Replace <path_to_directory> with the absolute path to your local pyviscous folder. 

Before using `pyviscous` this way, ensure its dependencies are installed manually:
```
pip install "wheel>=0.36"
pip install copulae
```

**Installation notes for Approach 3**  <br>
**(1) Why `wheel>=0.36` is needed?** <br>
- When the `wheel` package is not installed, `pip` falls back to the legacy `setup.py install` method to install `copulae` where dependencies like `numpy` are required at the build time.
- Since `copulae` uses a `pyproject.toml`-based build system, `pip` builds it in an isolated environment to prevent interference from your existing environment. This means it cannot access pre-installed packages like `numpy`, leading to the error.
- Installing `wheel>=0.36` enables pip to use the modern PEP 517/518 build process, which correctly handles build-time dependencies like `numpy` inside the isolated environment.<br>

**(2) Install `copulae` using `pip`, not `conda`.** <br>
The `conda` version of `copulae` lacks some source code and functionality, which can lead to runtime errors or missing features.<br>

**(3) Automatic dependencies installed by `copulae`.** <br>
When you install `copulae` (e.g., version 0.7.9), the following packages are installed automatically via pip:<br>
- numpy>=1.20
- pandas>=1.1
- scikit-learn>=1.2
- scipy>=1.5
- statsmodels>=0.12
- typing-extensions>=4.0.0
- wrapt>=1.12 <br>

You do **not** need to install these manually unless you want to control or pin their versions.

### Examples
We provide five demonstration notebooks in the example directory, including the Rosenbrock function and four Sobol’ functions from Liu et al. (2024). Additionally, a real case study of the **Bow at Banff basin, Alberta, Canada**, is included to show the real-world application of VISCOUS. Each example includes scripts for input-output data generation or reading, VISCOUS setup and execution, and evaluation of sensitivity results.

### Credits
VISCOUS was originally developed by Sheikholeslami et al. (2021) and enhanced by Liu et al. (2023). 
- Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes. *Water Resources Research*, 57, e2020WR028435. https://doi.org/10.1029/2020WR028435
- Liu, H., Clark, M. P., Gharari, S., Sheikholeslami, R., Freer, J., Knoben, W. J. M., Marsh C. B., & Papalexiou, S. M. (2024) An improved copula-based framework for efficient global sensitivity analysis. *Water Resources Research*, 60, e2022WR033808. https://doi.org/10.1029/2022WR033808

---
This package was created with Cookiecutter and the `https://github.com/audreyr/cookiecutter-pypackage` project template.
