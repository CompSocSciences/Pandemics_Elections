import os
import gc
import time
import math
import patsy
import random
import imageio
import graphviz
import datetime
# from dfply import *
import numpy as np
from numpy import mean,std,absolute
import pandas as pd
import seaborn as sns
from sklearn import tree
# import missingno as msno
# from pingouin import mwu
from pingouin import ancova
from patsy import dmatrices
# from ExKMC.Tree import Tree
import statsmodels.api as sm
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV,LinearRegression,ElasticNet,ElasticNetCV,Ridge, Lasso,RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error,r2_score,jaccard_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,RepeatedKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,normalize
from scipy.stats import ks_2samp,pearsonr,mannwhitneyu
from sklearn.cluster import KMeans
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE
from scipy.sparse import csc_matrix, find
from scipy.special import gammaln as lgamma
from scipy.signal import find_peaks, peak_widths
from statsmodels.genmod.families import Binomial
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_classification
from statsmodels.base.model import GenericLikelihoodModel
from imblearn.over_sampling import ADASYN, SMOTENC, BorderlineSMOTE
from scipy.stats import studentized_range

def median_distance(X):
    dist = distance.cdist(X, X, 'euclidean')
    h = np.median([dist[i, j] for i in range(len(X)) for j in range(len(X)) if i != j])
    return h
    
def rbf_kernel(X, Xp, h):    
    dist = distance.cdist(X, Xp, 'sqeuclidean')
    K = np.exp(- dist/2/h**2)    
    return K #n*m

def kernel_regression_fitting(xTrain, yTrain, h, beta=1):
    # X: input data, np array n*1
    # Y: input labels, np array n*1    
    K = rbf_kernel(xTrain, xTrain, h)
    W = np.dot(np.linalg.inv(K + beta*np.eye(len(K))), yTrain)    
    return W

def kernel_regression_fit_and_predict(xTrain, yTrain, xTest, h, beta):
    W = kernel_regression_fitting(xTrain, yTrain, h, beta)
    K_xTrain_xTest = rbf_kernel(xTrain, xTest, h)
    yPred = np.dot( K_xTrain_xTest.T, W)
    return yPred

def GetMaxFlow(flows):        
    maks=max(flows, key=lambda k: len(flows[k]))
    return len(flows[maks]), maks

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_kr_scatter(states, graph=True, type='cases', wl=7, beta=0.3, 
                    normalize=False,peak=True,gridix=[]):
    
    if len(gridix)==0:
        if len(states) == 2:
            gridix = [121,122]
        elif len(states) == 3:
            gridix = [221,222,223]
        elif len(states) == 4:
            gridix = [221,222,223,224]
        elif len(states) == 6:
            gridix = [231,234,232,235,233,236]

    max_yaxis = 0
    max_xaxis = 0
    for i,state in enumerate(states):
        covidst_plot = covidst[covidst.State== state]
        if type == 'cases':
            covidst_plot["daily"] = covidst_plot.cases_n.diff()
        else:
            covidst_plot["daily"] = covidst_plot.deaths_n.diff()
        covidst_plot.daily[covidst_plot.daily<0] = 0
        covidst_plot["ma"] = np.rint(covidst_plot.daily.rolling(window=wl).mean())
        covidst_plot = covidst_plot.iloc[wl:]
        if max(covidst_plot.ma*1.1) > max_yaxis:
            max_yaxis = max(covidst_plot.ma*1.1)
        if covidst_plot.index.shape[0]+wl > max_xaxis:
            max_xaxis = covidst_plot.index.shape[0]+wl        
    KR = {}
    peaks = {}
    plt.clf()    
    for i,state in enumerate(states):
        covidst_plot = covidst[covidst.State== state]
        pad_zeros = max_xaxis - covidst_plot.shape[0]
        zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covidst_plot.columns)
        covidst_plot = pd.concat([zero_df,covidst_plot])        
        if type == 'cases':
            covidst_plot["daily"] = covidst_plot.cases_n.diff()
        else:
            covidst_plot["daily"] = covidst_plot.deaths_n.diff()
        covidst_plot.daily[covidst_plot.daily<0] = 0
        covidst_plot["ma"] = np.rint(covidst_plot.daily.rolling(window=wl).mean())
        covidst_plot = covidst_plot.iloc[wl:]
        covidst_plot = covidst_plot.reset_index()
        covidst_plot['periods'] = covidst_plot.index            
        xTrain = covidst_plot.periods[:,np.newaxis]
        yTrain = covidst_plot.ma
        h = beta*median_distance(xTrain)
        yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTrain, h, beta)
        yHatk[yHatk<0]=0
        if normalize:
            yHatk = yHatk/sum(yHatk)
        KR[state] = yHatk    
        peaks_temp, _ = find_peaks(yHatk)
        if (np.argmax(yHatk) in peaks_temp) == False:
            peaks_temp = np.append(peaks_temp,np.argmax(yHatk))        
        peaks[state] = peaks_temp
        if graph:
            w, wh, lips, rips= peak_widths(yHatk, peaks_temp,rel_height=0.5)
            lips = lips.astype(int)
            rips = rips.astype(int)
            ax = plt.subplot(gridix[i])
            ax.set_title(state)
            plt.plot(xTrain, yTrain, '*')
            if peak:
                plt.plot(xTrain, yHatk, '-k')    
                plt.plot(peaks_temp, yHatk[peaks_temp], "x")
                plt.plot(lips, yHatk[lips], "*")
                plt.plot(rips, yHatk[rips], "*")        
            ax.set_ylim([0,max_yaxis])            
    if graph:
        plt.show()
    return KR, peaks  


# def plot_kr_scatter_county(counties, graph=True, type='cases', wl=7, beta=0.3, 
#                     normalize=False,peak=True,gridix=[]):
#     if len(gridix)==0:
#         if len(counties) == 2:
#             gridix = [121,122]
#         elif len(counties) == 3:
#             gridix = [221,222,223]
#         elif len(counties) == 4:
#             gridix = [221,222,223,224]
#         elif len(counties) == 6:
#             gridix = [231,234,232,235,233,236]

#     max_yaxis = 0
#     max_xaxis = 0
#     for i,ctySt in enumerate(counties):
#         covid_plot = covid[covid.ctySt== ctySt]
#         if type == 'cases':
#             covid_plot["daily"] = covid_plot.cases_n.diff()
#         else:
#             covid_plot["daily"] = covid_plot.deaths_n.diff()
#         covid_plot.daily[covid_plot.daily<0] = 0
#         covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
#         covid_plot = covid_plot.iloc[wl:]
#         if max(covid_plot.ma*1.1) > max_yaxis:
#             max_yaxis = max(covid_plot.ma*1.1)
#         if covid_plot.index.shape[0]+wl > max_xaxis:
#             max_xaxis = covid_plot.index.shape[0]+wl        
#     KR = {}
#     peaks = {}
#     plt.clf()    
#     for i,ctySt in enumerate(counties):
#         covid_plot = covid[covid.ctySt== ctySt]
#         pad_zeros = max_xaxis - covid_plot.shape[0]
#         zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covid_plot.columns)
#         covid_plot = pd.concat([zero_df,covid_plot])
#         if type == 'cases':
#             covid_plot["daily"] = covid_plot.cases_n.diff()
#         else:
#             covid_plot["daily"] = covid_plot.deaths_n.diff()
#         covid_plot.daily[covid_plot.daily<0] = 0
#         covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
#         covid_plot = covid_plot.iloc[wl:]
#         covid_plot = covid_plot.reset_index()
#         covid_plot['periods'] = covid_plot.index
#         xTrain = covid_plot.periods[:,np.newaxis]
#         yTrain = covid_plot.ma
#         h = beta*median_distance(xTrain)
#         yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTrain, h, beta)
#         yHatk[yHatk<0]=0
#         if normalize:
#             yHatk = yHatk/sum(yHatk)
#         KR[ctySt] = yHatk    
#         peaks_temp, _ = find_peaks(yHatk)
#         if (np.argmax(yHatk) in peaks_temp) == False:
#             peaks_temp = np.append(peaks_temp,np.argmax(yHatk))        
#         peaks[ctySt] = peaks_temp
#         if graph:
#             w, wh, lips, rips= peak_widths(yHatk, peaks_temp,rel_height=0.5)
#             lips = lips.astype(int)
#             rips = rips.astype(int)
#             ax = plt.subplot(gridix[i])
#             ax.set_title(ctySt)
#             plt.plot(xTrain, yTrain, '*')
#             if peak:
#                 plt.plot(xTrain, yHatk, '-k')    
#                 plt.plot(peaks_temp, yHatk[peaks_temp], "x")
#                 plt.plot(lips, yHatk[lips], "*")
#                 plt.plot(rips, yHatk[rips], "*")
#             ax.set_ylim([0,max_yaxis])            
#     if graph:
#         plt.show()
#     return KR, peaks  

def plot_kr_scatter_county(counties, graph=True, type='cases', wl=7, beta=0.3, 
                    normalize=False,peak=True,gridix=[],\
                    cutoff='2020-11-03',covid=''):
    if len(gridix)==0:
        if len(counties) == 2:
            gridix = [121,122]
        elif len(counties) == 3:
            gridix = [221,222,223]
        elif len(counties) == 4:
            gridix = [221,222,223,224]
        elif len(counties) == 6:
            gridix = [231,234,232,235,233,236]

    max_yaxis = 0
    max_xaxis = 0
    for i,ctySt in enumerate(counties):
        covid_plot = covid[covid.ctySt== ctySt]
        covid_plot = covid_plot[covid_plot.date < cutoff]
        if type == 'cases':
            covid_plot["daily"] = covid_plot.cases.diff()
        else:
            covid_plot["daily"] = covid_plot.deaths.diff()
        covid_plot.daily[covid_plot.daily<0] = 0
        covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
        covid_plot = covid_plot.iloc[wl:]
        if max(covid_plot.ma*1.1) > max_yaxis:
            max_yaxis = max(covid_plot.ma*1.1)
        if covid_plot.index.shape[0]+wl > max_xaxis:
            max_xaxis = covid_plot.index.shape[0]+wl        
    KR = {}
    peaks = {}
    plt.clf()    
    for i,ctySt in enumerate(counties):
        covid_plot = covid[covid.ctySt== ctySt]
        covid_plot = covid_plot[covid_plot.date < cutoff]
        pad_zeros = max_xaxis - covid_plot.shape[0]
        zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covid_plot.columns)
        covid_plot = pd.concat([zero_df,covid_plot])
        if type == 'cases':
            covid_plot["daily"] = covid_plot.cases.diff()
        else:
            covid_plot["daily"] = covid_plot.deaths.diff()
        covid_plot.daily[covid_plot.daily<0] = 0
        covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
        covid_plot = covid_plot.iloc[wl:]
        covid_plot = covid_plot.reset_index()
        covid_plot['periods'] = covid_plot.index
        xTrain = covid_plot.periods[:,np.newaxis]
        yTrain = covid_plot.ma
        h = beta*median_distance(xTrain)
        yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTrain, h, beta)
        yHatk[yHatk<0]=0
        if normalize:
            yHatk = yHatk/sum(yHatk)
        KR[ctySt] = yHatk    
        peaks_temp, _ = find_peaks(yHatk)
        if (np.argmax(yHatk) in peaks_temp) == False:
            peaks_temp = np.append(peaks_temp,np.argmax(yHatk))        
        peaks[ctySt] = peaks_temp
        if graph:
            w, wh, lips, rips= peak_widths(yHatk, peaks_temp,rel_height=0.5)
            lips = lips.astype(int)
            rips = rips.astype(int)
            ax = plt.subplot(gridix[i])
            ax.set_title(ctySt)
            plt.plot(xTrain, yTrain, '*')
            if peak:
                plt.plot(xTrain, yHatk, '-k')    
                plt.plot(peaks_temp, yHatk[peaks_temp], "x")
                plt.plot(lips, yHatk[lips], "*")
                plt.plot(rips, yHatk[rips], "*")
            ax.set_ylim([0,max_yaxis])            
    if graph:
        plt.show()
    return KR, peaks  

def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

"""
Beta regression for modeling rates and proportions.

References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.

Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""

# this is only need while #2024 is open.
class Logit(sm.families.links.Logit):

    """Logit tranform that won't overflow with large numbers."""

    def inverse(self, z):
        return 1 / (1. + np.exp(-z))

_init_example = """

    Beta regression with default of logit-link for exog and log-link
    for precision.

    >>> mod = Beta(endog, exog)
    >>> rslt = mod.fit()
    >>> print rslt.summary()

    We can also specify a formula and a specific structure and use the
    identity-link for phi.

    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = Beta.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                         dat, Z=Z, link_phi=identity())

    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:

    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> mod = Beta.from_formula('methylation ~ disease + age + gender + coverage', df, Z)
    >>> rslt = mod.fit()

"""

class Beta(GenericLikelihoodModel):

    """Beta Regression.

    This implementation uses `phi` as a precision parameter equal to
    `a + b` from the Beta parameters.
    """

    def __init__(self, endog, exog, Z=None, link=Logit(),
            link_phi=sm.families.links.Log(), **kwds):
        """
        Parameters
        ----------
        endog : array-like
            1d array of endogenous values (i.e. responses, outcomes,
            dependent variables, or 'Y' values).
        exog : array-like
            2d array of exogeneous values (i.e. covariates, predictors,
            independent variables, regressors, or 'X' values). A nobs x k
            array where `nobs` is the number of observations and `k` is
            the number of regressors. An intercept is not included by
            default and should be added by the user. See
            `statsmodels.tools.add_constant`.
        Z : array-like
            2d array of variables for the precision phi.
        link : link
            Any link in sm.families.links for `exog`
        link_phi : link
            Any link in sm.families.links for `Z`

        Examples
        --------
        {example}

        See Also
        --------
        :ref:`links`

        """.format(example=_init_example)
        assert np.all((0 < endog) & (endog < 1))
        if Z is None:
            extra_names = ['phi']
            Z = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in \
                        (Z.columns if hasattr(Z, 'columns') else range(1, Z.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names

        super(Beta, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_phi = link_phi
        
        self.Z = Z
        assert len(self.Z) == len(self.endog)

    def nloglikeobs(self, params):
        """
        Negative log-likelihood.

        Parameters
        ----------

        params : np.ndarray
            Parameter estimates
        """
        return -self._ll_br(self.endog, self.exog, self.Z, params)

    def fit(self, start_params=None, maxiter=100000, maxfun=5000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        """

        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()
                                 ).fit(disp=False).params
            start_params = np.append(start_params, [0.5] * self.Z.shape[1])

        return super(Beta, self).fit(start_params=start_params,
                                        maxiter=maxiter, maxfun=maxfun,
                                        method=method, disp=disp, **kwds)

    def _ll_br(self, y, X, Z, params):
        nz = self.Z.shape[1]

        Xparams = params[:-nz]
        Zparams = params[-nz:]

        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_phi.inverse(np.dot(Z, Zparams))
        # TODO: derive a and b and constrain to > 0?

        if np.any(phi <= np.finfo(float).eps): return np.array(-np.inf)

        ll = lgamma(phi) - lgamma(mu * phi) - lgamma((1 - mu) * phi) \
                + (mu * phi - 1) * np.log(y) + (((1 - mu) * phi) - 1) \
                * np.log(1 - y)

        return ll