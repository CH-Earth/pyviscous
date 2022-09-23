import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde, multivariate_normal
from sklearn.mixture import GaussianMixture

def define_GSA_variable_index(n_xvariables):
    """ 
    Create variable indices for which the variance-based sensitivity indices are estimated.
    GSA: global sensitivity analysis.
    
    Parameters
    -------
    n_xvariables: int. Total number of input variables (eg, parameters).
    
    Returns
    -------
    GSAIndex : list. List of indices of x variable groups to be evaluated. 
    eg, [[0]], or [[0],[1],[2]], or [[0,1],[0,2],[1,2]]. 
    
    Notes
    -------
    Interaction between varibales is disabled for now, thus [[1],[2],[3]].
    This code can be extended to explicitly calculate interaction effect 
    (eg, second-order, third-order sensitivity indices)."""
    
    GSAIndex = []
    for d in range(n_xvariables):
        GSAIndex.append([d]) # Index starts from zero following python syntax.
    return GSAIndex

def standardize_data(data):
    """ 
    Standradize random data into the standard normal distribution using the kernel density estimation.
    referenece: https://stackoverflow.com/questions/52221829/python-how-to-get-cumulative-distribution-function-for-continuous-data-values
    reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
     
    Parameters
    -------
    data: array, shape (n, 1) or (n,). Data array to be standardized.
    
    Returns
    -------
    cdf_data: output array, shape is the sames as data. CDF of data.
    z_data: array, shape is the sames as data. Standardized data."""    
    
    data_shape   = np.shape(data)
    data         = data.reshape((1,len(data)))
    
    # construct a kernel-density estimate using Gaussian kernels.
    kde          = gaussian_kde(data)
    
    # CDF function
    cdf_function = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))    
    cdf_data     = cdf_function(data)
    
    # calculate the cdf corresponding z_data in the standard normal distribution.
    z_data       = norm.ppf(cdf_data, loc=0, scale=1)
    
    # reformat shape the same as input data
    cdf_data     = cdf_data.reshape(data_shape)
    z_data       = z_data.reshape(data_shape)
    
    return cdf_data,z_data

def sample_from_data(data, n_samples):
    """ 
    Generate random samples based on the input data using the kernel density function.
     
    Parameters
    -------
    data: array, shape (n, 1) or (n,). Input data to build the kernel density function.
    n_samples : int. Number of samples to generate from the built kernel sensity function.
    
    Returns
    -------
    sample : array, shape (n_samples, 1). Randomly generated sample.
    z_sample: array, shape (n_samples, 1). Standardized sample values.
    z_sample_pdf: array, shape (n_samples, 1). PDF of standardized sample. """    

    data         = data.reshape((1,len(data)))
    kde          = gaussian_kde(data)                   # build the kernel density function.
    sample       = kde.resample(n_samples)              # generate samples.

    cdf_function = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    sampleCDF    = cdf_function(sample)                 # calcualte sample cdf in kde.
    
    z_sample     = norm.ppf(sampleCDF, loc=0, scale=1)  # calcualte inverse of cdf in the normal space.

    sample       = sample.T                             # reshape into (n_sample,1).
    z_sample     = z_sample.T                           # reshape into (n_sample,1).
    z_sample_pdf = norm.pdf(z_sample,0,1)               # z_sample's pdf in the normal space.
    
    return sample, z_sample, z_sample_pdf

def fit_GMM(x,y,n_components):
    """ Fit the Gaussian mixture model (GMM).
    reference: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    reference: https://cmdlinetips.com/2021/03/gaussian-mixture-models-with-scikit-learn-in-python/
    
    Parameters
    -------
    x: array, shape (n_samples, n_xvariables). Variable samples in normal space. 
    y: array, shape (n_samples, ). Response samples in normal space. 
    n_components: integer or string "optimal". The number of components used in GMM inference. 
                  If n_components is an integer, it will be used as the fixed user-provided number of components.
                  If n_components is "optimal", it will look for an optimal number of components in the range of [1, n_xvariables+20].
        
    Returns
    -------
    best_model: object. The best fitted GMM. 
    
    Notes
    -------
    In GMM fitting, combine all the variables of x and y, and treat them as multivariates of GMM. """    
    
    # Part 1.  Fit the input data with the Gaussian Mixture Model using different number of clusters.
    if len(np.shape(x)) == 1: # if shape(nSample,) -> (nSample,1)
        x = x.reshape(-1,1) 
        
    n_xvariables = np.shape(x)[1]
    
    if isinstance(n_components, str) and n_components=='optimal':
        n_components = np.arange(1, n_xvariables+20) # Note: +20 is hard coded. Total number of candidate GMMs.
    elif isinstance(n_components, int) and n_components>0:
        n_components = [n_components]
    else:
        print('The provided n_components does not meet the requirement: a string "optimal" or an iteger graeater than one.')
    
    # Combine all the variables of x and y, and treat them as multivariates of the Gaussian mixture model.
    data = np.concatenate((x,y), axis=1)
    
    # Fit GMMs with different number of clusters.
    models = [GaussianMixture(n,covariance_type='full',max_iter=1000,n_init=10).fit(data) \
              for n in n_components]
    
    # Part 2. Compute the BIC score for each model.
    gmm_model_comparisons = pd.DataFrame({"n_components" : n_components, "BIC" : [m.bic(data) for m in models]})

    # Part 3. Identify the minimum BIC score corresponding index.
    best_model_index = gmm_model_comparisons['BIC'].idxmin()
    
    # Part 4. Identify the optimal Gaussian mixture model.
    best_model = models[best_model_index]
    
    return best_model

def sample_from_GMM(gmm, n_samples):
    """
    Generate random samples from the fitted Gaussian mixture model (GMM).
    reference: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/mixture/_base.py#L396
    Note the scikit-learn built-in sample function is slightly changed here because scikit-learn fixes the random seed in sampling.
    
    'Full' means the components may independently adopt any position and shape. For other covariance types of GMM, please refer to:
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    https://stats.stackexchange.com/questions/326671/different-covariance-types-for-gaussian-mixture-models#:~:text=A%20Gaussian%20distribution%20is%20completely,all%20of%20which%20are%20ellipsoids.
    
    Parameters
    -------
    gmm: object. Fitted GMM.
    n_samples : int. Number of samples to generate.
    
    Returns
    -------
    X : array, shape (n_samples, n_xvariables). Randomly generated sample.
    y : array, shape (nsamples,). Component labels. """

    n_samples_comp = np.random.multinomial(n_samples, gmm.weights_)

    if gmm.covariance_type == 'full':
        X = np.vstack([
            np.random.multivariate_normal(mean, covariance, int(sample))
            for (mean, covariance, sample) in zip(
                gmm.means_, gmm.covariances_, n_samples_comp)])
    y = np.concatenate([np.full(sample, j, dtype=int)
                       for j, sample in enumerate(n_samples_comp)])

    return (X, y)

def calculate_GMM_conditional_pdf(multivariateData,gmm):
    ''' 
    Calculate the conditional pdf of y in the fitted Gaussian mixture model (GMM).
    
    Parameters
    -------
    multivariateData: matrix. [X,Y] values in normal space. shape (N2,Xvar_num+1).
    gmm : object. Fitted GMM.
    
    Returns
    -------
    yCondPDF : array, shape (N2,1). Conditional pdf of y, f(y|x) in the given GMM.
    
    Notes
    -------
    - There are two approaches of calculating conditional pdf of y. 
    - The presented method is faster.
    - The second method is based on Eqs 32-35 of Hu, Z. and Mahadevan, S., 2019. 
    - Probability models for data-driven global sensitivity analysis. Reliability Engineering & System Safety, 187, pp.40-57.
    '''
    
    # Get attributes of the fitted GMM and y
    gmmWeights     = gmm.weights_      # shape (n_components,)
    gmmMeans       = gmm.means_        # shape (n_components, n_variables). 
                                       # n_variables = n_feature in sklearn.mixture.GaussianMixture reference.
    gmmCovariances = gmm.covariances_  # (n_components, n_variables, n_variables) if covariance_type = ‘full’ (by default).    
    gmmNComponents = gmm.n_components  # number of components

    # Method: use the relationship f(y|x) = f(x,y)/f(x) 
    # step 1. calculate f(x,y), joint pdf of (x,y) of the fitted GMM.
    logProb    = gmm.score_samples(multivariateData)  # compute the log probability of multivariateData under the model.
    xyJointPDF = np.exp(logProb)                      # get the joint probability of multivariateData of GMM.
    
    # step 2. calculate f(x), marginal pdf of x in the the fitted GMM. shape(nComponent).
    xMarginalPDFCpnt = [multivariate_normal.pdf(multivariateData[0,0:-1], mean=gmmMeans[iComponent,:-1], 
                                                cov=gmmCovariances[iComponent,:-1,:-1]) for iComponent in range(gmmNComponents)] 
    xMarginalPDF     = sum(xMarginalPDFCpnt*gmmWeights)
    
    # step 3. calculate f(y|x), conditional pdf of y on x in the fitted GMM.    
    yCondPDF         = np.divide(xyJointPDF,xMarginalPDF)
    yCondPDF         = yCondPDF.reshape(-1,1)

    return yCondPDF

def VISCOUS(x,y,zx,zy,sensType,GSAIndex,N1,N2,n_components):
    """ 
    Gaussian Mixture Copula-Based Estimator for first-order and total-effect sensitivity indices.
    reference: https://scikit-learn.org/0.16/modules/generated/sklearn.mixture.GMM.html
    reference: https://stackoverflow.com/questions/67656842/cumulative-distribution-function-cdf-in-scikit-learn
    
    Parameters
    -------
    x: array, shape (n_samples, n_xvariables). X values in observation space. 
    y: array, shape (n_samples, 1). Y values in observation space. 
    zx: array, shape (n_samples, n_xvariables). Standardized X values in normal space. 
    zy: array, shape (n_samples, 1). Standardized Y values in normal space. 
    sensType: str. Type of Sensitivity index calculation. Two options: 'first', 'total'.
    GSAIndex: list. List of indices of x variable groups to be evaluated. eg, [[0]], or [[0],[1],[2]], or [[0,1],[0,2],[1,2]]. 
    N1: int. Number of Monte Carlo samples used for the outer loop. 
    N2: int. Number of Monte Carlo samplesused for the inner loop. 
    n_components: integer or string "optimal". The number of components used in GMM inference. 
                  If n_components is an integer, it will be used as the fixed user-provided number of components.
                  If n_components is "optimal", it will look for an optimal number of components in the range of [1, n_xvariables+20].
    
    Returns
    -------
    sensIndex     : array, shape (nGSAGroup,). Sensitivity index result of all evaluated variable groups.
    fitted_gmm_ls : list, length (nGSAGroup).  Best fitted GMM results of all evaluated variable groups.
    
    Notes
    -------
    - When fitting GMM, x (variable) and y (response) are combined to be the multivariates of GMM.
    - Equation number here is referred to the equations of paper: Liu. et al. (2022). 
    - pyVISCOUS: An open-source tool for computationally frugal global sensitivity analysis. """

    # Prepare: Calculate y variance and generate Monte Carlo y samples based on given y samples.
    # y is sampled here, not in the inner loop, because this can help avoid poor cdf-y extrapolation when y data samples are highly skwewed.
    varY                   = np.var(y)
    MC_y, MC_zy, MC_zy_pdf = sample_from_data(y, N2)   

    # Loop variable groups to calculate each group's sensitivity index. 
    # For example, x1's first-order sensitivity, x2's first-order sensitivity,.., x10's first-order sensitivity.
    if sensType == 'first':
        print('Calculating first-order sensitivity indices...')
    elif sensType == 'total':
        print('Calculating total-effect sensitivity indices...')        

    nGSAGroup = len(GSAIndex)           # total number of variable groups for sensitivity analysis
    sensIndex = np.zeros((nGSAGroup,))  # sensitivity index results for nGSAGroup 
    fitted_gmm_ls = []                  # a list of best fitted GMM for nGSAGroup
    
    for iGSA in range(nGSAGroup):
        print('--- variable group %s --- '%(GSAIndex[iGSA]))

        # (1) Identify to-be-evaluated data samples iGSA_zx 
        if sensType == 'first':
            iGSA_zx = zx[:,GSAIndex[iGSA]] # select column(s)     
        elif sensType == 'total':               
            iGSA_zx = np.delete(zx, GSAIndex[iGSA], axis=1) # drop column(s) 
        
        # (2) Build the GMM (gmm_pdf) by fitting Gaussian density components to zx and zy in normal space.
        print('--- fitting GMM...')
        fitted_gmm = fit_GMM(iGSA_zx, zy, n_components)
        fitted_gmm_ls.append(fitted_gmm)

        # Check convergency. If gmm is not converged, report it and go to the next varaible group. 
        if not (fitted_gmm.converged_):
            print("ERROR: GMM fitting is not converged.")
            continue

        # (3) Two-loop-based Monte Carlo approximations of the first-order and the total-order sensitivity indices. 
        print('--- calculating sensitivity index...')
        
        ########################################################################################
        # OUTER LOOP: Loop N1 Monte Carlo samples to compute V(E(y│x_i)) based on Eq 24.
        
        # --- Generate N1 Monte Carlos multivariable samples based on the fitted GMM (for the outer loop). 
        # sklearn has its built-in sample function, but its random seed is fixed.
        # The built-in sample function should not be used if users need to 
        # repeat the experiment multiple times to quantify the sampling uncertainty. 
        # MC_z1, MC_cpntLabel1 = fitted_gmm.sample(N1) 
        
        # here we sample with a non-fixed random seed.
        MC_z1, MC_cpntLabel1  = sample_from_GMM(fitted_gmm, N1)   # return MC_z1 = (Zx,Zy) and each sample corresponding component label.
        condEy = np.zeros((N1,1))                                 # conditional expectation of y given x
        
        # --- Loop N1 Monte Carlos samples.
        for iMC in range(N1):

            ############################################
            # INNER LOOP: Loop N2 Monte Carlo samples to compute E(y│x_i) based on Eq 21.
            
            # --- Construct N2 Monte Carlos multivariable samples.
            # (1) Get the iMC^th zx sample. Sample number is 1.
            iMC_zx        = MC_z1[iMC,0:-1]               
            # (2) Get the y samples. Sample number is N2.
            iMC_zy        = MC_zy.flatten()
            iMC_zy_pdf    = MC_zy_pdf
            iMC_y         = MC_y
            # (3) Construct 2nd GMM multivariable samples.
            MC_z2         = np.ones((N2,np.shape(MC_z1)[1]))    # create an array of shape (N2,n_variables), filled with ones.
            MC_z2[:,0:-1] = np.ones_like(MC_z2[:,0:-1])*iMC_zx  # fill zx variables with the iMC^th zx sample from MC_z1.
            MC_z2[:,-1]   = iMC_zy                              # fill zy variable with N2 previously generated Monte Carlo samples.

            # --- Loop N2 Monte Carlo samples using array operations. 
            # (1) Given zx, compute conditional pdf of zy, f(zy|zx)=fGMM(z)/fGMM(zx). 
            iMC_zy_gmmCondPDF = calculate_GMM_conditional_pdf(MC_z2, fitted_gmm)             
            # (2) Given x, compute conditional pdf of y, f(y|x)=f(zy|zx)/zy_pdf.
            iMC_y_gmmCondPDF  = iMC_zy_gmmCondPDF/iMC_zy_pdf           
            # (3) Given x, compute conditional expectation of y, E(y|x) (Eq 21). 
            iMC_condEy        = sum(iMC_y*iMC_y_gmmCondPDF)/float(N2)   
            
            # End INNER LOOP.
            ############################################

            # Save E(y|x).   
            condEy[iMC] = iMC_condEy             

        # Calculate Var(E(y|x)) (Eq 24).
        varCondEy = np.var(condEy) 
        # End OUTER LOOP.
        ########################################################################################

        # Calulcate sensitivity index.   
        if sensType == 'first':
            iGSA_s = varCondEy/varY          # (Eq 16)  
        elif sensType == 'total':
            iGSA_s = 1-varCondEy/varY        # (Eq 17)   
        
        print('--- Sensitivity index = %.6f'%(iGSA_s))
        sensIndex[iGSA] = iGSA_s             # save result.  
        
        print()
        
    return sensIndex, fitted_gmm_ls
