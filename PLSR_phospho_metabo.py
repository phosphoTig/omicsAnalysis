# -*- coding: utf-8 -*-
"""
@authors: TYT, RSM, FV
original code developed by RSM can be accessed via: https://github.com/RackS103/Omics_Analysis_Scripts
@machine: Luisa
"""

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.signal import savgol_filter
from sys import stdout

# from Enrichment_Scripts_fromRM import run_Enrichr, run_KEA3, run_STRING
# from PLSDA_fromRM import PLSClassifier

from mbpls.mbpls import MBPLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn import metrics
from natsort import natsorted
from scipy.stats import ttest_ind



def vip(model, feature_labels, X):
    """
    Regex_1 and Regex_2 is specific to Tig's project and is used to add in the Fold changes for two populations

    model <- PLSR model
    feature_labels <- columns of 
    X <- X phospho matrix
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape

    vips = np.zeros((p,))
    foldsF = np.zeros((p,))
    foldsM = np.zeros((p,))
    df=X.T
    FNCD = df.loc[:, [int(col[1:]) <= 19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ]
    FHFD=df.loc[:, [int(col[1:]) > 19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ]
    MNCD=df.loc[:, [int(col[1:]) <= 20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ]
    MHFD=df.loc[:, [int(col[1:]) > 20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ]

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        foldsF[i] = np.mean(FHFD.T.iloc[:, i]) / np.mean(FNCD.T.iloc[:, i])
        foldsM[i] = np.mean(MHFD.T.iloc[:, i]) / np.mean(MNCD.T.iloc[:, i])
        

    coef_col = 0
    vips =pd.concat([pd.DataFrame({'VIP': vips, 'Coef':model.coef_[:,coef_col], 
                        'F_Log2(HFD/NCD)': np.log2(foldsF), 'M_Log2(HFD/NCD)': np.log2(foldsM) }, index = feature_labels),df], axis=1)
    vips=vips.sort_values(by='VIP', ascending=False)
                        
    return vips


#this is for a single Y variable
def feature_selection(X, y, y_name, n):
    '''
    Selects the top N (default 50) features with the highest magnitude correlation to the given Y variable.
    
    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        y_name (str): Name of the target variable.
        n (int): Number of top features to select.
    
    Returns:
        DataFrame: Selected features.
    '''
    # Calculate correlation coefficients between each feature and the target variable
    corr_values = X.apply(lambda col: np.corrcoef(col, y)[0, 1])
    
    # Create DataFrame to store correlation coefficients
    corr_df = pd.DataFrame({'Feature': X.columns, 'Correlation': corr_values.abs()})
    
    # Select top N features based on absolute correlation values
    selected_features = corr_df.nlargest(n, 'Correlation')['Feature']
    
    # Return selected features
    return X[selected_features]

def loocv_score_singleY(model, scorer, X, Y):
    """
    Q^2 score for univariate Y matrix.
    model <- sklearn model
    scorer <- scoring function
    X <- X matrix, pandas format only
    Y <- Y matrix, should be a 1D vector/pandas Series.
    """
    loo = LeaveOneOut()
    Y_hat_test = np.zeros(Y.shape)
    train_scores = []
    latent_variable_scores = []
    percent_vars_explained = []
    loadings = []
    
    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx, :]
        X_test =  X[test_idx, :]
        Y_train = Y[train_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model.fit(X_train, Y_train)
        Y_hat_train = model.predict(X_train)
        train_scores.append(scorer(Y_train, Y_hat_train))
        Y_hat_test[test_idx] = model.predict(X_test)
        
        # Extract latent variable scores
        latent_variable_scores.append(model.transform(X_test)[:, :2])  # Assuming 2 latent variables
        
        # Calculate variance explained for the first two components
        u, s, _ = np.linalg.svd(X_train, full_matrices=False)
        explained_variance = (s**2) / np.sum(s**2)
        percent_vars_explained.append(explained_variance[:2] * 100)  # First two components
        
        # Extract loadings
        loadings.append(model.x_loadings_[:, :2])  # Assuming 2 latent variables
        
    # Calculate Q²
    press = np.sum((Y - Y_hat_test) ** 2)
    tss = np.sum((Y - np.mean(Y)) ** 2)
    q2 = 1 - press / tss
    
    # Calculate RMSE and MAE
    rmse = mean_squared_error(Y, Y_hat_test, squared=False)
    mae = mean_absolute_error(Y, Y_hat_test)
    
    # Create a dataframe for the results
    LV_df = pd.DataFrame({
        'Percent Variance Explained': percent_vars_explained,
        'Latent Variable Scores': latent_variable_scores,
        'Loadings': loadings
    })
    
    return np.mean(train_scores), scorer(Y, Y_hat_test), LV_df, q2, rmse, mae



def PLS_CV(X, Y, model_class=PLSRegression, gs_scoring='neg_mean_squared_error', score_fx=r2_score, cv_range=np.arange(2,20,2), multi_Y=False, verbose=False):
    """
    X <- phospho matrix
    Y <- Metabolomics matrix
    model_class <- type of model to use (PLSRegression or PLSClassifier)
    gs_scoring <- sklearn scoring function string name, used in GridSearchCV to find optimal n_components for model
    score_fx <- function to score the performance of model.
    cv_range <- range of values to test for n_components in PLSR
    multi_Y <- True if Y matrix is multivariate
    verbose <- prints out results as function works if True
    """
    
    
    
    if multi_Y: 
        Y = StandardScaler().fit_transform(Y)
    else:
        Y = stats.zscore(Y.astype(float))
    print(model_class())
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('predictor', model_class())])
    gs = GridSearchCV(estimator=pipe, param_grid={'predictor__n_components':cv_range}, 
                      cv=LeaveOneOut(), scoring=gs_scoring)
    gs.fit(X, Y)

    ncomp = gs.best_params_['predictor__n_components']
    model = model_class(n_components=ncomp)
    
    model.fit(X, Y)
    y_c = model.predict(X)
 
    # Cross-val
    y_cv = cross_val_predict(model, X, Y, cv=LeaveOneOut())
 
    # Calibration scores and errors
    r2_c = r2_score(Y, y_c)
    rmse_c = mean_squared_error(Y, y_c, squared=False)
    mae_c = mean_absolute_error(Y, y_c)
 
    # Cross-val scores and errors
    r2_cv = r2_score(Y, y_cv)
    rmse_cv = mean_squared_error(Y, y_cv, squared=False)
    mae_cv = mean_absolute_error(Y, y_cv)
 
    if verbose:
        print(f'Best model was {model} with {gs_scoring} {gs.best_score_}')
        print(f'Calibration R²: {r2_c}, Calibration RMSE: {rmse_c}, Calibration MAE: {mae_c}')
        print(f'Cross-validated R²: {r2_cv}, Cross-validated RMSE: {rmse_cv}, Cross-validated MAE: {mae_cv}')

    if multi_Y:
        train_score, test_score = loocv_score_multiY(model, score_fx, X.to_numpy(), Y)
    else:
        train_score, test_score, LVs, q2, rmse, mae = loocv_score_singleY(model, score_fx, X.to_numpy(), Y)
        
    if verbose:
        print(f'Train Performance ({score_fx.__name__}): {train_score}')
        print(f'Test Performance ({score_fx.__name__}): {test_score}')
        print(f'Predictive Ability (Q²): {q2}')
        print(f'Cross-validated RMSE: {rmse}')
        print(f'Cross-validated MAE: {mae}')
    
    return model, r2_c, rmse_c, mae_c, q2, rmse_cv, mae_cv,ncomp, LVs



def kfold_score_singleY(model, scorer, X, Y):
    """
    Q^2 score for univariate Y matrix.
    model <- sklearn model
    scorer <- scoring function
    X <- X matrix, pandas format only
    Y <- Y matrix, should be a 1D vector/pandas Series.
    """
    loo = KFold(n_splits=7)
    Y_hat_test = np.zeros(Y.shape)
    train_scores = []
    latent_variable_scores = []
    percent_vars_explained = []
    loadings = []
    
    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx, :]
        X_test =  X[test_idx, :]
        Y_train = Y[train_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model.fit(X_train, Y_train)
        Y_hat_train = model.predict(X_train)
        train_scores.append(scorer(Y_train, Y_hat_train))
        
        # Ensure predictions are 1D
        Y_hat_test[test_idx] = model.predict(X_test).flatten()
        
        # Extract latent variable scores
        latent_variable_scores.append(model.transform(X_test)[:, :2])  # Assuming 2 latent variables
        
        # Calculate variance explained for the first two components
        u, s, _ = np.linalg.svd(X_train, full_matrices=False)
        explained_variance = (s**2) / np.sum(s**2)
        percent_vars_explained.append(explained_variance[:2] * 100)  # First two components
        
        # Extract loadings
        loadings.append(model.x_loadings_[:, :2])  # Assuming 2 latent variables
        
    # Calculate Q²
    press = np.sum((Y - Y_hat_test) ** 2)
    tss = np.sum((Y - np.mean(Y)) ** 2)
    q2 = 1 - press / tss
    
    # Calculate RMSE and MAE
    rmse = mean_squared_error(Y, Y_hat_test, squared=False)
    mae = mean_absolute_error(Y, Y_hat_test)
    
    # Create a dataframe for the results
    LV_df = pd.DataFrame({
        'Percent Variance Explained': percent_vars_explained,
        'Latent Variable Scores': latent_variable_scores,
        'Loadings': loadings
    })
    
    return np.mean(train_scores), scorer(Y, Y_hat_test), LV_df, q2, rmse, mae

def kfold_PLS_CV(X, Y, model_class=PLSRegression, gs_scoring='neg_mean_squared_error', score_fx=r2_score, cv_range=np.arange(1, 40, 1), multi_Y=False, verbose=False):
    """
    X <- phospho matrix
    Y <- Metabolomics matrix
    model_class <- type of model to use (PLSRegression or PLSClassifier)
    gs_scoring <- sklearn scoring function string name, used in GridSearchCV to find optimal n_components for model
    score_fx <- function to score the performance of model.
    cv_range <- range of values to test for n_components in PLSR
    multi_Y <- True if Y matrix is multivariate
    verbose <- prints out results as function works if True
    """
    
    if multi_Y: 
        Y = StandardScaler().fit_transform(Y)
    else:
        Y = stats.zscore(Y.astype(float))
    
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('predictor', model_class())])
    
    # First stage grid search
    gs = GridSearchCV(estimator=pipe, param_grid={'predictor__n_components':cv_range}, 
                      cv=KFold(n_splits=7), scoring=gs_scoring)
    gs.fit(X, Y)

    # Refine around the best parameter found in the first stage
    best_ncomp = gs.best_params_['predictor__n_components']
    refined_range = np.arange(max(1, best_ncomp-5), best_ncomp+6)
    
    gs_refined = GridSearchCV(estimator=pipe, param_grid={'predictor__n_components':refined_range}, 
                              cv=KFold(n_splits=7), scoring=gs_scoring)
    gs_refined.fit(X, Y)
    
    ncomp = gs_refined.best_params_['predictor__n_components']
    model = model_class(n_components=ncomp)
    
    model.fit(X, Y)
    y_c = model.predict(X)
 
    # Cross-val
    y_cv = cross_val_predict(model, X, Y, cv=KFold(n_splits=7))
 
    # Calibration scores and errors
    r2_c = r2_score(Y, y_c)
    rmse_c = mean_squared_error(Y, y_c, squared=False)
    mae_c = mean_absolute_error(Y, y_c)
 
    # Cross-val scores and errors
    r2_cv = r2_score(Y, y_cv)
    rmse_cv = mean_squared_error(Y, y_cv, squared=False)
    mae_cv = mean_absolute_error(Y, y_cv)
 
    if verbose:
        print(f'Best model was {model} with {gs_scoring} {gs.best_score_}')
        print(f'Calibration R²: {r2_c}, Calibration RMSE: {rmse_c}, Calibration MAE: {mae_c}')
        print(f'Cross-validated R²: {r2_cv}, Cross-validated RMSE: {rmse_cv}, Cross-validated MAE: {mae_cv}')

    if multi_Y:
        train_score, test_score = loocv_score_multiY(model, score_fx, X.to_numpy(), Y)
    else:
        train_score, test_score, LVs, q2, rmse, mae = loocv_score_singleY(model, score_fx, X.to_numpy(), Y)
        
    if verbose:
        print(f'Train Performance ({score_fx.__name__}): {train_score}')
        print(f'Test Performance ({score_fx.__name__}): {test_score}')
        print(f'Predictive Ability (Q²): {q2}')
        print(f'Cross-validated RMSE: {rmse}')
        print(f'Cross-validated MAE: {mae}')
    
    return model, r2_c, rmse_c, mae_c, q2, rmse_cv, mae_cv,ncomp, LVs





def do_analysis(X, Y, name, expType, classifier):
    model, r2_c, rmse_c, mae_c, q2, rmse_cv, mae_cv,ncomp, LVs= PLS_CV(X,Y)
    
    newData['Name'].append(name)
    newData['Q2'].append(q2)
    newData['R2'].append(r2_c)
    newData['RMSE_cv'].append(rmse_cv)
    newData['MAE_cv'].append(mae_cv)
    newData['RMSE_training'].append(rmse_c)
    newData['MAE_training'].append(mae_c)
    newData['Model'].append(model)
    newData['Components'].append(ncomp)
    
    
    if q2>=0.4:
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
    
        vip = np.zeros((p,))
        foldsF = np.zeros((p,))
        foldsM = np.zeros((p,))
        pvalsF = np.zeros((p,))
        pvalsM = np.zeros((p,))
        df=X.T
        if "BHA" in expType:
        
            FNCD = df.loc[:, [int(col[2:]) <= 5 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ].T
            FHFD=df.loc[:, [int(col[2:]) > 5 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ].T
            MNCD=df.loc[:, [int(col[2:]) <= 10 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ].T
            MHFD=df.loc[:, [int(col[2:]) > 10 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ].T
        
        else:
            FNCD = df.loc[:, [int(col[1:]) <= 19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ].T
            FHFD=df.loc[:, [int(col[1:]) > 19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True] for col in df.columns] ].T
            MNCD=df.loc[:, [int(col[1:]) <= 20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ].T
            MHFD=df.loc[:, [int(col[1:]) > 20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True] for col in df.columns] ].T
        
        
    
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
    
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vip[i] = np.sqrt(p*(s.T @ weight)/total_s)
            foldsF[i] = np.mean(FHFD.iloc[:, i]) / np.mean(FNCD.iloc[:, i])
            t, pvalsF[i] = ttest_ind(FNCD.iloc[:,i], FHFD.iloc[:,i])
            foldsM[i] = np.mean(MHFD.iloc[:, i]) / np.mean(MNCD.iloc[:, i])
            t, pvalsM[i] = ttest_ind(MNCD.iloc[:,i], MHFD.iloc[:,i])
       
    
        coef_col = 0
        vips[name] = pd.concat([pd.DataFrame({'VIP': vip, 'Coef':model.coef_[:,coef_col],'F_Log2(HFD/NCD)': np.log2(foldsF), 'M_Log2(HFD/NCD)': np.log2(foldsM) }, index = X.columns),df], axis=1)
        
    
    return newData, vips, LVs



# read in all phospho and metabo data for first chorot: HFD v NCD
Xm = pd.read_csv('NCDvHFD_70mc-Metabolomics.csv', index_col='Metabolites')
Xm = Xm.fillna(1).T
Xm_f=Xm.iloc[0:38,:]
Xm_m=Xm.iloc[38:,:]


Xy= pd.read_csv('NCDvHFD70_PRM_new_clusterTable.csv', index_col='Label')
Xy=Xy.drop(axis=1, labels=['Clusters']) 
Xy=Xy.reindex(columns=natsorted(Xy.columns))
Xy= Xy.T
Xy_f=Xy.iloc[0:38,:]
Xy_m=Xy.iloc[38:,:]




#read in all phospho & metabo data for BHA chorot: HFD v HFD+BHA
Xm = pd.read_csv('20240429_BHA_70.csv', index_col='Class_Metabolites').drop(['Class','Metabolites'], axis=1)
Xm = Xm.fillna(0).T
Xm_f=Xm.iloc[0:10,:]
Xm_m=Xm.iloc[10:,:]

Xy=pd.read_csv('BHA_PRM_pY_brgA_u_wClass.csv', index_col=['Label']).drop(['extras','Structural_annot','Class', 'Sites'], axis=1)
Xy=Xy.fillna(1)
Xy= Xy.T
Xy_f=Xy.iloc[0:10,:]
Xy_m=Xy.iloc[10:,:]







# Data dictionary
newData = {'Name':[], 'Q2':[],'R2':[],'RMSE_cv':[],'MAE_cv':[],'RMSE_training':[],'MAE_training':[], 'Model':[], 'Components':[]}
vips={}
LVs_dict={}



# Perform PLSR on single Y with or without feature selection
for i in range(len(Xm_f.columns)):
    expType="BHA"
    phenos = Xm_f.columns
    name = phenos[i]

    # Identify common rows
    common_index = Xm_f.index.intersection(Xy_f.index)

    # Select features present in both dataframes
    X_sel = Xy_f.loc[common_index]
    # Select features present in both dataframes
    y= Xm_f.loc[common_index]
    
    
    # Perform feature selection: if running with no FS then use n=len(X_sel.T), for top 50% of co-correlated features use n=len(X_sel.T)//2
    X_sel = feature_selection(X_sel, y.loc[common_index, name], name, n=len(X_sel.T)//2)

    # Perform analysis with selected features
    newData, vips, LVs_dict[name] = do_analysis(X_sel,y[name], name, expType, classifier=False)



# Create DataFrame from selected keys
newDataframe = pd.DataFrame({key: newData[key] for key in ['Name', 'Q2','R2','RMSE_cv','MAE_cv', 'RMSE_training','MAE_training', 'Model', 'Components']})


newDataframe=newDataframe.sort_values(by='Q2',ascending=False)
newDataframe.to_csv("F_PRM_BHA_top50p_PLSR.csv")

PLSRhits=newDataframe.loc[newDataframe['Q2']>=0.4, 'Name'].tolist()

directory_name = "F_PRM_BHA_top50p_VIPs"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)

for k, v in vips.items():
    if k in PLSRhits:
        k=k.replace('/','_')
        v=v.sort_values(by='VIP', ascending=False)
        filename = f"{k}_VIPs.csv"
        filepath = os.path.join(directory_name, filename)
        v.to_csv(filepath, index=True)


directory_name = "F_PRM_BHA_top50p_LVs"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)

for k, v in LVs_dict.items():
    if k in PLSRhits:
        k=k.replace('/','_')
        filename = f"{k}_LVs.csv"
        filepath = os.path.join(directory_name, filename)
        v.to_csv(filepath, index=True)