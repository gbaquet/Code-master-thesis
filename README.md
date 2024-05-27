Description of the files:
- mod1.py and mod2.py implement the extrapolation method from chapter 5. It uses functions from the file loc_lin_est.pyx (Cython file) to implement the local linear estimator. In order to run the Python files, you will need to compile the Cython file by running the following line in the terminal: python setup.py build_ext --inplace
- cross_val.py and sim_cv.py are the files used and the end of chapter 4. The first one shows the estimation of a covariance function when using a cross-validation method or the theoretical sum of squares. sim_cv.py shows how the chosen bandwidths compare between the two methods. cross_val.py uses functions from cross_val.pyx whereas sim_cv.py uses opt_cv.pyx .
To compile cross_val.pyx : python setup_cv.py build_ext --inplace
To compile cross_val.pyx : python setup_opt.py build_ext --inplace
