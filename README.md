# ML_opt
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
[PyPi project link](https://pypi.org/project/mloptima/0.0.1/)
## Repository of optimization methods in machine learning.
## Project structure
```bash
├── classification
│   ├── __init__.py
│   └── models.py
├── gradients
│   ├── __init__.py
│   ├── optimize.py
│   ├── plotting.py
│   └── README.md
├── __init__.py
├── integerprogramming
│   ├── __init__.py
│   ├── optimize.py
│   └── utils.py
├── interiorpoint
│   ├── __init__.py
│   ├── optimize.py
│   └── plotting.py
├── onedimensionaloptimization
│   ├── __init__.py
│   ├── optimize.py
│   ├── plotting.py
│   └── utils_onedimensionaloptimization.py
├── regression
│   ├── __init__.py
│   └── models.py
├── stochasticoptimization
│   ├── __init__.py
│   ├── models.py
│   ├── optimize.py
│   └── plotting.py
└── twovarextremas
    ├── extrema_methods.py
    ├── find_extrema.py
    ├── __init__.py
    ├── plotting_3d.py
    └── utils_twovarextremas.py
```
## Machine Learning models and valuable visualization.
1. Two-variable function extrema finding and visualization implementation. Use User interface notebook twovarextremas Jupyter Notebook with downloaded modules.
2. One dimensional optimization(minimization) methods implementation and visualization. Use User interface notebook onedimensionaloptimization Jupyter Notebook with downloaded modules.
3. Gradients optimization(minimization) methods and visualization implementation. Use User interface notebook gradients Jupyter Notebook with downloaded modules.
4. Regression(Linear, Polynomial, Exponential) models and visualization implementation with StochasticGradientDescent/NormalEquation solvers. Use User interface notebook regression colab Jupyter Notebook with downloaded modules.
5. Optimization with equality/inequality constraints. Interior point methods and visualization implementation. Use User interface notebook interiorpoint Jupyter Notebook with downloaded modules.
6. Classification. Implementation of LogisticRegression(Ridge, Lasso), LogisticRegression with RBF kernel function, SVMs classifiers. Use User Interface notebook classification notebook colab with downloaded modules.
7. Integer linear programming. Implementation of Tabular Simplex method and Gomori cutting plane method for integer linear programming.
8. Stochastic optimization. Implementation of Stochastic gradient descent, Support vector classifier on 2 classes optimized by SGD, simulated annealing algorithm for function minimization, genetic algorithm for function minimization. Built in visualization implemented.
