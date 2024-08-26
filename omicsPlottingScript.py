# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:50:53 2021

@author: tytamir
"""
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import gcf
import seaborn as sns
sns.set(font="Arial")
plt.rcParams['pdf.fonttype']=42

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from collections import defaultdict

from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
import fastcluster as fc
from matplotlib.colors import rgb2hex, colorConverter

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import StandardScaler
#from bioinfokit.visuz import cluster0
from functools import partial, reduce
from scipy.stats import ttest_ind, pearsonr, false_discovery_control, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.mixture import GaussianMixture

from adjustText import adjust_text
from matplotlib.colors import ListedColormap

#if there are treament and control groups, you can define the colors you would like to use for the heatmap based on column names, make sure to call this function in the heatmap function
def plottingColors (df):
    #Define treatment and conrol groups, replicates, and timepoints. Store in dicitionary for color coding plots. edit this for different experiments
    plt.rcParams['pdf.fonttype']=42
    palette=sns.color_palette("Paired",11)
    palette2=sns.color_palette('Spectral',11)
    palette3=sns.color_palette('RdBu',11)
    F=[col for col in df.columns if col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    M=[col for col in df.columns if col in df.columns[df.columns.str.contains('|'.join(['M']))==True]] 
    B=[col for col in df.columns if col in df.columns[df.columns.str.contains('|'.join(['B\\d']))==True]]
    FNCD=[col for col in df.drop(B,axis=1).columns if int(col[1:])<=19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    FHFD=[col for col in df.drop(B,axis=1).columns if int(col[1:])>19 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    MNCD=[col for col in df.drop(B,axis=1).columns if int(col[1:])<=20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True]]
    MHFD=[col for col in df.drop(B,axis=1).columns if int(col[1:])>20 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True]]
    color_dict={}
    for col in df:
        if col in F:
            color_dict[col]=palette[8]
        elif col in M:
            color_dict[col]=palette[6]
        elif col in B:
            color_dict[col]=palette[10]
    labs1=pd.Series(color_dict).to_frame()
    #Define treatment and conrol groups, replicates, and timepoints. Store in dicitionary for color coding plots. edit this for different experiments
    for col in df:
        if col in FNCD:
            color_dict[col]=palette[2]
        elif col in MNCD:
            color_dict[col]=palette[2]
        elif col in FHFD:
            color_dict[col]=palette[4]
        elif col in MHFD:
            color_dict[col]=palette[4]
        elif col in B:
            color_dict[col]=palette[10]
    labs2=pd.Series(color_dict).to_frame()
	
    labs=pd.concat(((labs2.rename(columns={labs2.columns[0]:'Diet'})),(labs1.rename(columns={labs1.columns[0]:'Sex'}))),axis=1)
    
    return labs

#if there are treament and control groups, you can define the colors you would like to use for the heatmap based on column names, make sure to call this function in the heatmap function
def plottingColors2 (df):
    #Define treatment and conrol groups, replicates, and timepoints. Store in dicitionary for color coding plots. edit this for different experiments
    plt.rcParams['pdf.fonttype']=42
    palette=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])
    F=[col for col in df.columns if col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    M=[col for col in df.columns if col in df.columns[df.columns.str.contains('|'.join(['M']))==True]] 
    FHFD=[col for col in df.columns if int(col[2:])<=5 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    FBHA=[col for col in df.columns if int(col[2:])>5 and col in df.columns[df.columns.str.contains('|'.join(['F']))==True]]
    MHFD=[col for col in df.columns if int(col[2:])<=10 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True]]
    MBHA=[col for col in df.columns if int(col[2:])>10 and col in df.columns[df.columns.str.contains('|'.join(['M']))==True]]
    color_dict={}
    for col in df:
        if col in F:
            color_dict[col]=palette[3]
        elif col in M:
            color_dict[col]=palette[2]

    labs1=pd.Series(color_dict).to_frame()
    #Define treatment and conrol groups, replicates, and timepoints. Store in dicitionary for color coding plots. edit this for different experiments
    for col in df:
        if col in FHFD:
            color_dict[col]=palette[0]
        elif col in MHFD:
            color_dict[col]=palette[0]
        elif col in FBHA:
            color_dict[col]=palette[1]
        elif col in MBHA:
            color_dict[col]=palette[1]

    labs2=pd.Series(color_dict).to_frame()
	
    labs=pd.concat(((labs2.rename(columns={labs2.columns[0]:'BHA'})),(labs1.rename(columns={labs1.columns[0]:'Sex'}))),axis=1)
    
    return labs


#hirearchically clustered heatmap that can be saved in a pdf format. 
def heatmapsANDcorrelation(data, filename, labs, zeroval,metric, yclust,export):
    df=data.copy()
    df=df.fillna(zeroval)
    # Prepare a vector of color mapped to the 'label' column
    # retrieve clusters using fcluster 
    d = sch.distance.pdist(df)
    L = sch.linkage(d, metric=metric,method='complete')
    #picked 5 clusters
    clusters = sch.fcluster(L, 5, criterion='maxclust', monocrit=None)
    
    
    #save cluster assignments to dictionary by using df[Label] column as key
    tempClusters={}
    for i,cluster in enumerate(clusters):
        tempClusters[df.index[i]]= cluster
    
    hexColorpalette=['#fd5956','#0cb577','#ffbacd','#00ffff','#7b002c','#f9bc08','#d0fe1d','#b0054b','#0c1793']
    mycolors=sns.set_palette(sns.color_palette(hexColorpalette))
    #map cluster assignments back to the re-indexed dataframe
    # df=df.reset_index()
    df['Cluster']=df.index.map(tempClusters)
    row_colors = df['Cluster'].map(dict(zip(list(np.unique(df['Cluster'])),sns.color_palette(mycolors,max(list(df['Cluster']))))))
    
    myplot_noL=sns.clustermap(df.drop(['Cluster'],axis=1),metric=metric, method='complete',robust=True,cmap=sns.blend_palette(["#0165fc", "1", "#be0119"], 9),vmin=0,center=1,col_colors=labs,row_colors=row_colors, row_cluster=True, col_cluster=yclust, yticklabels=False,xticklabels=True)
    # Set the super title for the entire figure
    myplot_noL.fig.suptitle(f"Heatmap for {filename}")
    outheatmap=myplot_noL.data2d
    outheatmap['Clusters']=outheatmap.index.map(tempClusters)
    if export==True:
        myplot_noL.savefig(filename+'_new.pdf')
        outheatmap.to_csv(filename+'_new_clusterTable.csv')
	
	
    
	#spearman correlation heatmap of peptides, can be exproted to pdf
    dataT=df.drop('Cluster', axis=1).T
    dataTcorr=dataT.corr('spearman')
    dR = sch.distance.pdist(dataTcorr)
    LR = sch.linkage(dR, metric='euclidean',method='complete')
    # 5 clusters
    clusters = sch.fcluster(LR, 5, criterion='maxclust', monocrit=None)
    
    #save cluster assignments to dictionary by using df[Label] column as key
    tempClustersR={}
    for i,cluster in enumerate(clusters):
        tempClustersR[dataTcorr.index[i]]= cluster
    
    
    dataTcorr['Cluster']=dataTcorr.index.map(tempClustersR)
    row_colorsR = dataTcorr['Cluster'].map(dict(zip(list(np.unique(dataTcorr['Cluster'])),sns.color_palette(mycolors,max(list(dataTcorr['Cluster']))))))
    
    rowcorrs_noL=sns.clustermap(dataTcorr.drop('Cluster',axis=1),metric='euclidean', method='complete',vmin=-1, vmax=1, row_colors=row_colorsR, col_colors=row_colorsR, robust=True,cmap=sns.blend_palette(["#0165fc", "1", "#be0119"], 9), yticklabels=False,xticklabels=False)
    # Set the super title for the entire figure
    rowcorrs_noL.fig.suptitle(f"Peptide cluster map for {filename}")
    outrowcorr=rowcorrs_noL.data2d
    outrowcorr['Clusters']=outrowcorr.index.map(tempClustersR)
    if export==True:
        outrowcorr.to_csv(filename+'_new_rowCorr.csv')
        rowcorrs_noL.savefig(filename+'_new_rowCorr.pdf')    
    
    
    #sample correlation 
    corr=df.drop('Cluster',axis=1).corr('spearman')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI Symbol','simHei','Arial','sans-serif']
    #define the significant r value cuttoffs, can add this label to heatmap if you want to label significant correlations
    sigVals = corr.apply(lambda x: x.map(lambda val: u'\u2605' if val > 0.6 or val < -0.6 else ''))
    sampleCorr=sns.clustermap(corr,metric='euclidean', method='complete', vmin=-1,col_colors=labs,row_colors=labs,vmax=1,robust=True,cmap='RdBu_r', yticklabels=True, xticklabels=True)
    # Set the super title for the entire figure
    sampleCorr.fig.suptitle(f"Spearman Correlation for {filename}")
    outsamplecorr=sampleCorr.data2d
    if export==True:
        outsamplecorr.to_csv(filename+'_Samplecorr.csv')
        sampleCorr.savefig(filename+'_Samplecorr.pdf')
    
    return outheatmap, outrowcorr, outsamplecorr


def pval_n_log2fc(data, name, fc_cutoff, nanVals):
    df = data.copy()
    if 'BHA' in name:
        FNCD = [col for col in df.columns if int(col[2:]) <= 5 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        FHFD = [col for col in df.columns if int(col[2:]) > 5 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        MNCD = [col for col in df.columns if int(col[2:]) <= 10 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        MHFD = [col for col in df.columns if int(col[2:]) > 10 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        results_dfM = pd.DataFrame(columns=[name+'_M_aveHFD', name+'_M_aveBHA', name+'_M_pValue', name+'_M_-Log10(FDR)', name+'_M_Log2(BHA/HFD)', name+'_M_sig'])
        results_dfF = pd.DataFrame(columns=[name+'_F_aveHFD', name+'_F_aveBHA',name+'_F_pValue', name+'_F_-Log10(FDR)', name+'_F_Log2(BHA/HFD)', name+'_F_sig'])
        
    else:
        FNCD = [col for col in df.columns if int(col[1:]) <= 19 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        FHFD = [col for col in df.columns if int(col[1:]) > 19 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        MNCD = [col for col in df.columns if int(col[1:]) <= 20 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        MHFD = [col for col in df.columns if int(col[1:]) > 20 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        results_dfM = pd.DataFrame(columns=[name+'_M_aveNCD', name+'_M_aveHFD', name+'_M_pValue', name+'_M_-Log10(FDR)', name+'_M_Log2(HFD/NCD)', name+'_M_sig'])
        results_dfF = pd.DataFrame(columns=[name+'_F_aveNCD', name+'_F_aveHFD',name+'_F_pValue', name+'_F_-Log10(FDR)', name+'_F_Log2(HFD/NCD)', name+'_F_sig'])
        
    dfM = df.filter(like='M')
    threshM = 0.70 * len(dfM.columns)
    dfM = dfM.dropna(thresh=threshM).fillna(nanVals)
    
    p_values_M = []
    for index, row in dfM.iterrows():
        groupsM = []
        for col in [MNCD, MHFD]:
            groupsM.append(row[col].values)
        
        _, p_val = ttest_ind(*groupsM, alternative='two-sided')
        p_values_M.append(p_val)
    
    p_values_M = np.array(p_values_M)
    pvals_corrected_M = multipletests(p_values_M, method='fdr_bh')[1]

    idx = 0
    for index, row in dfM.iterrows():
        groupsM = []
        for col in [MNCD, MHFD]:
            groupsM.append(row[col].values)

        log_fc = math.log2(groupsM[1].mean() / groupsM[0].mean())
        sig = 'ns'
        if pvals_corrected_M[idx] < 0.05 and abs(log_fc) >= fc_cutoff:
            sig = 'pos' if log_fc >= fc_cutoff else 'neg'
        
        
        results_dfM.loc[index] = [groupsM[0].mean(), groupsM[1].mean(), p_values_M[idx], -np.log10(pvals_corrected_M[idx]), log_fc, sig]
        idx += 1
    

    dfF = df.filter(like='F')
    threshF = 0.70 * len(dfF.columns)
    dfF = dfF.dropna(thresh=threshF).fillna(nanVals)
    
    p_values_F = []
    for index, row in dfF.iterrows():
        groupsF = []
        for col in [FNCD, FHFD]:
            groupsF.append(row[col].values)

        _, p_val = ttest_ind(*groupsF, alternative='two-sided')
        p_values_F.append(p_val)
        
    p_values_F = np.array(p_values_F)
    pvals_corrected_F = multipletests(p_values_F, method='fdr_bh')[1]

    idx = 0
    for index, row in dfF.iterrows():
        groupsF = []
        for col in [FNCD, FHFD]:
            groupsF.append(row[col].values)

        log_fc = math.log2(groupsF[1].mean() / groupsF[0].mean())
        sig = 'ns'
        if pvals_corrected_F[idx] < 0.05 and abs(log_fc) >= fc_cutoff:
            sig = 'pos' if log_fc >= fc_cutoff else 'neg'
        
        results_dfF.loc[index] = [groupsF[0].mean(), groupsF[1].mean(), p_values_F[idx], -np.log10(pvals_corrected_F[idx]), log_fc, sig]
        idx += 1

    return results_dfM, results_dfF



def pval_n_log2fc(data, name, fc_cutoff, nanVals):
    df = data.copy()
    if 'BHA' in name:
        FNCD = [col for col in df.columns if int(col[2:]) <= 5 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        FHFD = [col for col in df.columns if int(col[2:]) > 5 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        MNCD = [col for col in df.columns if int(col[2:]) <= 10 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        MHFD = [col for col in df.columns if int(col[2:]) > 10 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        results_dfM = pd.DataFrame(columns=[name+'_M_aveHFD', name+'_M_aveBHA', name+'_M_pValue', name+'_M_-Log10(FDR)', name+'_M_Log2(HFD/BHA)', name+'_M_sig'])
        results_dfF = pd.DataFrame(columns=[name+'_F_aveHFD', name+'_F_aveBHA',name+'_F_pValue', name+'_F_-Log10(FDR)', name+'_F_Log2(HFD/BHA)', name+'_F_sig'])
        
    else:
        FNCD = [col for col in df.columns if int(col[1:]) <= 19 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        FHFD = [col for col in df.columns if int(col[1:]) > 19 and col in df.columns[df.columns.str.contains('|'.join(['F'])) == True]]
        MNCD = [col for col in df.columns if int(col[1:]) <= 20 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        MHFD = [col for col in df.columns if int(col[1:]) > 20 and col in df.columns[df.columns.str.contains('|'.join(['M'])) == True]]
        results_dfM = pd.DataFrame(columns=[name+'_M_aveNCD', name+'_M_aveHFD', name+'_M_pValue', name+'_M_-Log10(FDR)', name+'_M_Log2(HFD/NCD)', name+'_M_sig'])
        results_dfF = pd.DataFrame(columns=[name+'_F_aveNCD', name+'_F_aveHFD',name+'_F_pValue', name+'_F_-Log10(FDR)', name+'_F_Log2(HFD/NCD)', name+'_F_sig'])
        
    dfM = df.filter(like='M')
    threshM = 0.70 * len(dfM.columns)
    dfM = dfM.dropna(thresh=threshM).fillna(nanVals)
    
    p_values_M = []
    for index, row in dfM.iterrows():
        groupsM = []
        for col in [MNCD, MHFD]:
            groupsM.append(row[col].values)
        
        _, p_val = ttest_ind(*groupsM, alternative='two-sided')
        p_values_M.append(p_val)
    
    p_values_M = np.array(p_values_M)
    pvals_corrected_M = multipletests(p_values_M, method='fdr_bh')[1]

    idx = 0
    for index, row in dfM.iterrows():
        groupsM = []
        for col in [MNCD, MHFD]:
            groupsM.append(row[col].values)

        log_fc = math.log2(groupsM[0].mean() / groupsM[1].mean())
        sig = 'ns'
        if pvals_corrected_M[idx] < 0.05 and abs(log_fc) >= fc_cutoff:
            sig = 'pos' if log_fc >= fc_cutoff else 'neg'
        
        
        results_dfM.loc[index] = [groupsM[0].mean(), groupsM[1].mean(), p_values_M[idx], -np.log10(pvals_corrected_M[idx]), log_fc, sig]
        idx += 1
    

    dfF = df.filter(like='F')
    threshF = 0.70 * len(dfF.columns)
    dfF = dfF.dropna(thresh=threshF).fillna(nanVals)
    
    p_values_F = []
    for index, row in dfF.iterrows():
        groupsF = []
        for col in [FNCD, FHFD]:
            groupsF.append(row[col].values)

        _, p_val = ttest_ind(*groupsF, alternative='two-sided')
        p_values_F.append(p_val)
        
    p_values_F = np.array(p_values_F)
    pvals_corrected_F = multipletests(p_values_F, method='fdr_bh')[1]

    idx = 0
    for index, row in dfF.iterrows():
        groupsF = []
        for col in [FNCD, FHFD]:
            groupsF.append(row[col].values)

        log_fc = math.log2(groupsF[0].mean() / groupsF[1].mean())
        sig = 'ns'
        if pvals_corrected_F[idx] < 0.05 and abs(log_fc) >= fc_cutoff:
            sig = 'pos' if log_fc >= fc_cutoff else 'neg'
        
        results_dfF.loc[index] = [groupsF[0].mean(), groupsF[1].mean(), p_values_F[idx], -np.log10(pvals_corrected_F[idx]), log_fc, sig]
        idx += 1

    return results_dfM, results_dfF







#Mean center and z-score the data to get fold change over the average value per row (i.e. per phosphosite)
def meanCenter_data(df,eT, export):
    df=df.assign(ave=df.loc[:,df.columns].mean(axis=1))
    dfmc=df.loc[:,df.columns].div(df['ave'], axis=0)
    dfmc=dfmc.drop(['ave'], axis=1)
    df=df.assign(stndev=df.loc[:,df.columns].std(axis=1))
    dfzs=(df.loc[:,df.columns].sub(df['ave'], axis=0)).div(df['stndev'], axis=0)
    dfzs=dfzs.drop(['ave','stndev'], axis=1)
    
    if export==True:
        dfmc.to_csv(eT+'mc.csv')

    return dfmc, dfzs



#Import pre-processed files ready for plotting for targeted phosphoproteomics runs
data=pd.read_csv('FILENAME.csv', sep=',')
data['Gene']=data['Gene'].str.replace('GN=','')
data['Label']=data['Gene']+'_'+data['protSite']
pY_PRM=data.set_index('Label').drop(['Gene','protSite'], axis=1).rename(columns={'M38b':'M39'})


#remove rows with less than 70% NAN columns
dropThresh = 0.70 * len(pY_PRM.columns)
pY_PRM_70=pY_PRM.dropna(thresh=dropThresh)


#Import pre-processed files ready for plotting for untargeted phosphoproteomics runs
data=pd.read_csv('FILENAME.csv', sep=',')
data['Class_Label']=data['Class']+'-'+data['Label']
pY_DDA=data.set_index('Class_Label').drop(['Gene','protSite','Label','Class','Class_full'], axis=1).rename(columns={'M38b':'M39'})

#remove rows with less than 70% NAN columns
dropThresh = 0.70 * len(pY_DDA.columns)
pY_DDA_70=pY_DDA.dropna(thresh=dropThresh)

#Import pre-processed files ready for plotting: metabolomics runs
metDF=pd.read_csv('FILENAME.csv', sep=',')
metDF['Class_Metabolites']=metDF['Class']+'_'+metDF['Metabolites']
metDF=metDF.set_index(['Class_Metabolites']).drop(['Class','Metabolites'], axis=1)

dropThresh = 0.70 * len(metDF.columns)
metDF_70=metDF.dropna(thresh=dropThresh)

#mean center metabolomics dataset
metDF_70mc, _ =meanCenter_data(metDF_70, 'metDF_70', export=True)



#generate heatmap for dataframes of interest, see below example
clustermapDict={}
clustermapDict['pY_PRM_clustermap'],clustermapDict['pY_PRM_peptCorr'], clustermapDict['pY_PRM_sampleCorr']=heatmapsANDcorrelation(pY_PRM_70,'pY_PRM', plottingColors(pY_PRM_70), zeroval=1,metric='euclidean',yclust=True, export=True)


#generate log2 fold change analysis for males and females separately for dataframes of interest, see below example
volcdict={}
volcdict['pY_PRM_M'], volcdict['pY_PRM_F']=pval_n_log2fc(pY_PRM_70,'pY_PRM', 0.6, 0)


#Generate volcanoplots and save as PDF
for k, v in volcdict.items():
    sns.set_theme(style='ticks')
    sns.scatterplot(data=v, x=k+'_Log2(HFD/NCD)', y=k+'_-Log10(FDR)', hue=v[k+'_sig'].apply(str),palette={'pos':'red', 'neg':'blue', 'ns':'silver'},size=k+'_-Log10(FDR)', legend=False)
    plt.xlabel('Log2(HFD/NCD)')
    plt.ylabel('-Log10(FDR)')
    # Define the cutoff for significant points
    
    plt.axhline(1.3, color='silver', linestyle='--',linewidth=0.5)
    plt.axvline(-0.6, color='silver', linestyle='--',linewidth=0.5)
    plt.axvline(0.6, color='silver', linestyle='--',linewidth=0.5)
    plt.title(k)
    # sig_labels = v[abs(v['Log2(HFD/NCD)'])>=0.6].nlargest(100, '-Log10(p-Value)').index.tolist()
    sig_labels_positive = v[v[k+'_-Log10(FDR)'] >= 1.3].nlargest(20, k+'_Log2(HFD/NCD)').index.tolist()
    sig_labels_negative = v[v[k+'_-Log10(FDR)'] >= 1.3].nsmallest(15, k+'_Log2(HFD/NCD)').index.tolist()

    # Combine both lists
    sig_labels = sig_labels_positive + sig_labels_negative


    for i in sig_labels:
        plt.text(v.loc[i, k+'_Log2(HFD/NCD)'], v.loc[i, k+'_-Log10(FDR)'], i, ha='center', va='center', fontsize=4)
    # plt.savefig(k+'_volcplot.pdf')
    # plt.clf()
    plt.show()


for i, j in volcdict.items():
    j.to_csv(i+"_2_volcplot.csv")






#prepare fold change dataframe for plotting
AveFC_pY_PRM=pd.concat([volcdict['pY_PRM_M'], volcdict['pY_PRM_F']], axis=1)
AveFC_pY_PRM=AveFC_pY_PRM.reset_index()
AveFC_pY_PRM['Class']=AveFC_pY_PRM['index'].str.split('_', expand=True)[0]
AveFC_pY_PRM['Label'] = AveFC_pY_PRM['index'].str.split('_', expand=True)[1]
AveFC_pY_PRM=AveFC_pY_PRM.drop('index', axis=1)



# Define the classes to plot
classes = AveFC_pY_PRM['Class'].unique()

# Create a KDE plot for each class (pathway) specific phosphosites or metabolites
for cls in classes:
    subset = AveFC_pY_PRM[AveFC_pY_PRM['Class'] == cls]
    
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(subset['pY_PRM_M_Log2(HFD/NCD)'], label='Male', fill=True, color='blue', alpha=0.5)
    sns.kdeplot(subset['pY_PRM_F_Log2(HFD/NCD)'], label='Female', fill=True, color='red', alpha=0.5)
    
    plt.title(f'KDE Plot for Class {cls}')
    plt.xlabel('Log2(HFD/NCD)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.show()






# Map class labels to numerical values for coloring

# Define the classes of metabolite pathway to plot
target_classes_met = ['Pur-D', 'Pur-S', 'Pyr-D', 'Pyr-S','CARB', 'Gly','PPP','TCA','RX','1CM','CF','AA','UC', 'acAA','FAO']

# Define the classes of pY modified enzymes pathway to plot
target_classes_pY = ['Pur','Pyr','Gly','CARB','Rx', 'CF', '1CM','TSS','UC','AAD','TCA','OXP', 'FA','AA', 'Mis']

# Merge 'AA' and 'AAD' into 'AA Catabolism', and label other classes as 'other'
AveFC_pY_PRM['Class'] = AveFC_pY_PRM['Class'].replace(['AA', 'AAD'], 'AA Cat')
AveFC_pY_PRM['Class'] = AveFC_pY_PRM['Class'].apply(lambda x: x if x in target_classes else 'other')

# Filter the DataFrame to include only the target classes and 'other'
filtered_df = AveFC_pY_PRM[AveFC_pY_PRM['Class'].isin(target_classes + ['other'])]

# Set the desired figure size
figsize = (5, 7)

# Order the 'Class' column based on the target_classes list
filtered_df['Class'] = pd.Categorical(filtered_df['Class'], categories=target_classes + ['other'], ordered=True)

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

# Set the color palette
palette = sns.color_palette('tab20c', len(target_classes) + 1)


# Create violin plot for females
sns.violinplot(x='pY_PRM_F_Log2(HFD/NCD)', y='Class', data=filtered_df, ax=axes[0], palette=palette, inner=None, scale='width', width=0.8)

# Plot non-significant points
sns.stripplot(x='pY_PRM_F_Log2(HFD/NCD)', y='Class', data=filtered_df[filtered_df['pY_PRM_F_sig'] == 'ns'],
              ax=axes[0], color='black', size=4, jitter=True, dodge=True)

# Plot significant points
sns.stripplot(x='pY_PRM_F_Log2(HFD/NCD)', y='Class', data=filtered_df[filtered_df['pY_PRM_F_sig'].isin(['neg', 'pos'])],
              ax=axes[0], color='yellow', edgecolor='black', size=4, jitter=True, dodge=True)

axes[0].set_title('Females')
axes[0].set_xlabel('Log2(HFD/NCD)')
axes[0].set_ylabel('')
axes[0].set_xlim(-5, 3)
axes[0].tick_params(axis='both', which='both', left=True, bottom=True)

# Create violin plot for males
sns.violinplot(x='pY_PRM_M_Log2(HFD/NCD)', y='Class', data=filtered_df, ax=axes[1], palette=palette, inner=None, scale='width', width=0.8)

# Plot non-significant points
sns.stripplot(x='pY_PRM_M_Log2(HFD/NCD)', y='Class', data=filtered_df[filtered_df['pY_PRM_M_sig'] == 'ns'],
              ax=axes[1], color='black', size=4, jitter=True, dodge=True)

# Plot significant points
sns.stripplot(x='pY_PRM_M_Log2(HFD/NCD)', y='Class', data=filtered_df[filtered_df['pY_PRM_M_sig'].isin(['neg', 'pos'])],
              ax=axes[1], color='yellow', edgecolor='black', size=4, jitter=True, dodge=True)

axes[1].set_title('Males')
axes[1].set_xlabel('Log2(HFD/NCD)')
axes[1].set_ylabel('')
axes[1].set_xlim(-5, 3)
axes[1].tick_params(axis='both', which='both', left=False, bottom=True)

# Remove the y-axis from the males plot
axes[1].yaxis.set_visible(False)

plt.tight_layout()
# plt.show()

plt.savefig('pY_PRM_M_log2_vPlots_byClass.pdf')




sns.set_theme(style='white')

# Map class labels to numerical values for coloring

# Define the classes of metabolite pathway to plot
selected_classes_met = ['Pur-D', 'Pur-S', 'Pyr-D', 'Pyr-S','CARB', 'Gly','PPP','TCA','RX','1CM','CF','AA','UC', 'acAA','FAO']

# Define the classes of pY modified enzymes pathway to plot
selected_classes_pY = ['Pur','Pyr','Gly','CARB','Rx', 'CF', '1CM','TSS','UC','AAD','TCA','OXP', 'FA','AA', 'Mis']

# Filter the DataFrame to include only the selected classes
filtered_df = AveFC_pY_PRM[AveFC_pY_PRM['Class'].isin(selected_classes)]

# Map class labels to numerical values for coloring
class_labels = filtered_df['Class'].tolist()
labeled_classes = list(set(class_labels))
class_to_num = {cls: i for i, cls in enumerate(labeled_classes)}

# Create a color palette that can handle all classes
mypalette = sns.color_palette("tab20c", len(labeled_classes))
palette = sns.color_palette(mypalette, len(labeled_classes))
cmap = ListedColormap(palette)

# Add color information to the DataFrame
filtered_df['color'] = filtered_df['Class'].map(lambda cls: class_to_num.get(cls))

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Set the desired figure size
figsize = (12, 8)

# Create the plot
fig, axes = plt.subplots(2, len(selected_classes), figsize=figsize, sharey='row')

for i, cls in enumerate(selected_classes):
    male_subset = filtered_df[filtered_df['Class'] == cls]
    female_subset = filtered_df[filtered_df['Class'] == cls]
    
    sns.kdeplot(male_subset['pY_PRM_M_Log2(HFD/NCD)'], ax=axes[0, i], bw_adjust=1, fill=True, alpha=1, linewidth=1.5, color=palette[i])
    sns.kdeplot(male_subset['pY_PRM_M_Log2(HFD/NCD)'], ax=axes[0, i], clip_on=False, color="w", lw=2, bw_adjust=1)
    axes[0, i].set_title(f'Male - {cls}')
    axes[0, i].set_xlim(-2.5, 2.5)
    axes[0, i].spines['left'].set_visible(False)
    axes[0, i].spines['bottom'].set_visible(False)
    axes[0, i].set_xlabel('')  # Remove x-axis label for the first row
    if i == 0:
        axes[0, i].set_ylabel('Density')

    sns.kdeplot(female_subset['pY_PRM_F_Log2(HFD/NCD)'], ax=axes[1, i], bw_adjust=1, fill=True, alpha=1, linewidth=1.5, color=palette[i])
    sns.kdeplot(female_subset['pY_PRM_F_Log2(HFD/NCD)'], ax=axes[1, i], clip_on=False, color="w", lw=2, bw_adjust=1)
    axes[1, i].set_title(f'Female - {cls}')
    axes[1, i].set_xlim(-2.5, 2.5)
    axes[1, i].spines['left'].set_visible(False)
    axes[1, i].spines['bottom'].set_visible(False)
    axes[1, i].set_xlabel('Log2(HFD/NCD)')  # Add x-axis label for the bottom row
    if i == 0:
        axes[1, i].set_ylabel('Density')

plt.tight_layout()
plt.show()













#Generate KDE plots 
# ax = sns.kdeplot(FC_met['BHA70mc_F_Log2(HFD/NCD)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[3], fill=True, linewidth=2, alpha=0.9)
# ax = sns.kdeplot(FC_met['BHA70mc_M_Log2(HFD/NCD)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[2], fill=True, linewidth=2, alpha=0.6)
ax = sns.kdeplot(AveFC_pY_PRM['pY_PRM_M_Log2(HFD/BHA)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[2], fill=True, linewidth=0, alpha=0.9)
ax = sns.kdeplot(AveFC_pY_PRM['pY_PRM_F_Log2(HFD/BHA)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[3], fill=True, linewidth=0, alpha=0.6)
# ax = sns.kdeplot(AveFC_gpBHA70['BHA_gProt70_F_Log2(BHA/HFD)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[3], fill=True, linewidth=2, alpha=0.9)
# ax = sns.kdeplot(AveFC_gpBHA70['BHA_gProt70_M_Log2(BHA/HFD)'], color=sns.color_palette(['#F5846D','#1A85FF','#FFE56A','#E1CDF7'])[2], fill=True, linewidth=2, alpha=0.6)
# ax.set_xlim(-1,2)
# ax.set_ylim(0,3.25)
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Average Abundance', font='Arial',fontsize=15)
ax.set_ylabel('Density of pY', font='Arial', fontsize=15,labelpad=15)
# ax.set_xlabel('Log2(HFD/BHA)', font='Arial',fontsize=15)
# ax.set_ylabel('Density of Metabolites', font='Arial', fontsize=15,labelpad=15)
labels = [ 'M','F']
ax.legend(labels, loc='best', fontsize='medium')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='x', labelsize=15)
ax.set_facecolor('none')
ax.tick_params(axis='both', which='both', left=True, bottom=True)
for tick in ax.get_xticklabels():
    tick.set_fontname('Arial')
for tick in ax.get_yticklabels():
    tick.set_fontname('Arial')

plt.savefig('pY_PRM_kde_.pdf', bbox_inches='tight')





#------------------------------------------------------------------------------------------------------------------------------------------------------------
#PCA of samples and calculation of variance 
def PCA_analysis(df,eT,colorlabs, classes,nloadings,export):
    dataT=df.T.reset_index()
    x = dataT.iloc[:,1:]
    y = dataT.loc[:,['index']]
    x_st=StandardScaler().fit_transform(x)
    
    normX=pd.DataFrame(x.T,columns=y['index'])
    pca_out=PCA().fit(x)
    loadings=pca_out.components_
    num_pc=pca_out.n_features_
    
    cumsums=np.cumsum(pca_out.explained_variance_ratio_)
    
    pc_list = ["PC"+str(i) for i in list(range(1, 15))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = y['index']
    loadings_df = loadings_df.set_index('variable')
    
    pca_scores=PCA().fit_transform(x)
    
    pca = PCA(n_components=5)
    components = pca.fit_transform(x)
    
    total_var = pca.explained_variance_ratio_.sum() * 100
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    
    num_components = min(5, components.shape[1])
    sns.set_theme(style='ticks')
    
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i + 1}' for i in range(components.shape[1])], index=df.T.columns)
    
    
    
    # Create a grid of plots for the first 5 PCs
    fig, axes = plt.subplots(num_components, num_components, figsize=(15, 15))

    for i in range(num_components):
        for j in range(num_components):
            if i == j:
                sns.histplot(components[:, i], kde=True, ax=axes[i, j])  
                axes[i, j].set_title(f'PC{i+1}')
            else:
                axes[i, j].scatter(components[:, j], components[:, i], c=colorlabs, marker='o', edgecolor=classes)
                axes[i, j].set_xlabel(f'PC{j+1} ('+ str(round(pca_out.explained_variance_ratio_[j] * 100, 2)) + '%)')
                axes[i, j].set_ylabel(f'PC{i+1} ('+ str(round(pca_out.explained_variance_ratio_[i] * 100, 2)) + '%)')

    # Adjust layout
    plt.tight_layout()
    
    if export==True:
        plt.savefig(eT + '_PCA_grid_plots.pdf', bbox_inches='tight')
        plt.show()
        plt.clf()
    else:
        plt.show()
    
    
    # Identify top-contributing features for each PC
    top_contributors = {}
    for col in loadings_df.columns:
        top_contributors[col] = loadings_df[col].abs().nlargest(nloadings).index.tolist()
        
    # Create loading plots for all combinations of the first 4 PCs
    for i in range(4):
        for j in range(i + 1, 4):
            plt.figure(figsize=(6, 8))
            plt.scatter(loadings_df[f'PC{i + 1}'], loadings_df[f'PC{j + 1}'], marker='o', c='gray', alpha=0.5)
            for k, row in loadings_df.iterrows():
                plt.plot([0, row[f'PC{i + 1}']], [0, row[f'PC{j + 1}']], color='gray', linestyle='--', linewidth=0.25)
            plt.xlabel(f'Loadings PC{i + 1}')
            plt.ylabel(f'Loadings PC{j + 1}')
            plt.title(f'Loading Plot for PC{i + 1} and PC{j + 1}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            
            # Annotate top contributors for each PC combination
            texts = []
            labeled_features = set()  # To keep track of already labeled features
            for feature in top_contributors[f'PC{i + 1}']:
                if feature in loadings_df.index and feature not in labeled_features:
                    texts.append(
                        plt.text(loadings_df.loc[feature, f'PC{i + 1}'], loadings_df.loc[feature, f'PC{j + 1}'], feature,
                                 ha='right', va='bottom', color='blue', fontsize=8))
                    plt.scatter(loadings_df.loc[feature, f'PC{i + 1}'], loadings_df.loc[feature, f'PC{j + 1}'], marker='o', c='orangered')
                    labeled_features.add(feature)  # Add the feature to the set of labeled features
                    
            for feature in top_contributors[f'PC{j + 1}']:
                if feature in loadings_df.index and feature not in labeled_features:
                    texts.append(
                        plt.text(loadings_df.loc[feature, f'PC{i + 1}'], loadings_df.loc[feature, f'PC{j + 1}'], feature,
                                 ha='right', va='bottom', color='blue', fontsize=8))
                    plt.scatter(loadings_df.loc[feature, f'PC{i + 1}'], loadings_df.loc[feature, f'PC{j + 1}'], marker='o', c='orangered')
                    labeled_features.add(feature)  # Add the feature to the set of labeled features
            # Adjust the position and orientation of labels for optimal readability
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), force_text=(0.1, 0.2), autoalign='y')
            
            
            # Export the figure
            if export==True:
                plt.savefig(f'{eT}_PC{i + 1}_PC{j + 1}_loadings.pdf', bbox_inches='tight')
                plt.show()
                plt.clf()
                
            else:
                plt.show()
            
        
        return loadings_df

  

colorlabs1=[plottingColors(pY_PRM_70).to_dict()['Sex'][a] for a in pY_PRM_70.columns]
classes1=[plottingColors(pY_PRM_70).to_dict()['Diet'][a] for a in pY_PRM_70.columns]
pY_PRM_70_loadings=PCA_analysis(pY_PRM_70.fillna(1),'pY_PRM_70',colorlabs1,classes1, nloadings=25, export=True) 
    





def PCA_analysis_byClass(df, eT, colorlabs, classes, nloadings, mypalette, export, labeled_classes):
    dataT = df.T.reset_index()
    x = dataT.iloc[:, 1:]
    y = dataT.loc[:, ['index']]
    x_st = StandardScaler().fit_transform(x)
    
    normX = pd.DataFrame(x.T, columns=y['index'])
    pca_out = PCA().fit(x)
    loadings = pca_out.components_
    num_pc = pca_out.n_features_
    
    cumsums = np.cumsum(pca_out.explained_variance_ratio_)
    
    pc_list = ["PC" + str(i) for i in list(range(1, 15))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = y['index']
    loadings_df = loadings_df.set_index('variable')
    
    pca_scores = PCA().fit_transform(x)
    
    pca = PCA(n_components=5)
    components = pca.fit_transform(x)
    
    total_var = pca.explained_variance_ratio_.sum() * 100
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    
    num_components = min(5, components.shape[1])
    sns.set_theme(style='ticks')
    
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i + 1}' for i in range(components.shape[1])], index=df.T.columns)
    
    
    # Map class labels to numerical values for coloring
    class_labels = df.index.get_level_values(0).tolist()
    if labeled_classes is None:
        labeled_classes = list(set(class_labels))
    class_to_num = {cls: i for i, cls in enumerate(labeled_classes)}
    
    # Create a color palette that can handle all classes
    num_colors = len(labeled_classes)
    palette = sns.color_palette(mypalette, num_colors)
    cmap = ListedColormap(palette)

    # Add color information to loadings_df
    loadings_df['class'] = df.index.get_level_values(0)
    loadings_df['color'] = loadings_df['class'].map(lambda cls: class_to_num.get(cls))

    # Identify top-contributing features for each PC
    top_contributors = {}
    for col in loadings_df.columns[:-2]:  # exclude 'class' and 'color' columns
        top_contributors[col] = loadings_df[col].abs().nlargest(nloadings).index.tolist()
        
    # Create loading plots for the first 4 PCs
    for i in range(4):
        for j in range(i + 1, 4):
            plt.figure(figsize=(6, 8))
            scatter = plt.scatter(loadings_df[f'PC{i + 1}'], loadings_df[f'PC{j + 1}'], marker='o', c=loadings_df['color'], cmap=cmap, alpha=1)
            for k, row in loadings_df.iterrows():
                plt.plot([0, row[f'PC{i + 1}']], [0, row[f'PC{j + 1}']], color='gray', linestyle='--', linewidth=0.25)
            plt.xlabel(f'Loadings PC{i + 1}')
            plt.ylabel(f'Loadings PC{j + 1}')
            plt.title(f'Loading Plot for PC{i + 1} and PC{j + 1}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
    
            # Add a legend outside the plot
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls, 
                                  markersize=10, markerfacecolor=palette[class_to_num[cls]]) for cls in labeled_classes]
            plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
            if export:
                plt.savefig(eT + f'_PC{i + 1}_PC{j + 1}_loadings.pdf', bbox_inches='tight')
                plt.show()
                plt.clf()
            else:
                plt.show()
                plt.clf()
        
    return loadings_df





# Define the classes to plot
labeled_classes_met = ['Pur-D', 'Pur-S', 'Pyr-D', 'Pyr-S','CARB', 'Gly','PPP','TCA','RX','1CM','CF','AA Cat','UC', 'acAA','FAO','other']

# Merge 'AA' and 'AAD' into 'AA Catabolism', and label other classes as 'other'
testDF=metDF_70.copy().reset_index()
testDF['Class']=testDF['Class_Metabolites'].str.split('_', expand=True)[0]
testDF['Metabolites'] = testDF['Class_Metabolites'].str.split('_', expand=True)[1]
testDF=testDF.drop('Class_Metabolites', axis=1)
testDF['Class'] = testDF['Class'].replace(['AA', 'AAD'], 'AA Cat')
testDF['Class'] = testDF['Class'].apply(lambda x: x if x in labeled_classes else 'other')
testDF=testDF.set_index(['Class','Metabolites']) 

 
colorlabs1=[plottingColors2(testDF).to_dict()['Sex'][a] for a in testDF.columns]
classes1=[plottingColors2(testDF).to_dict()['BHA'][a] for a in testDF.columns]




labeled_classes_pY = ['Pur','Pyr','Gly','CARB','Rx', 'CF', '1CM','TSS','UC','AAD','TCA','OXP', 'FA','AA', 'Mis']

testDF=pY_PRM_70.copy().reset_index()
testDF['Class']=testDF['Class_Label'].str.split('_', expand=True)[0]
testDF['Label'] = testDF['Class_Label'].str.split('_', expand=True)[1]
testDF=testDF.drop('Class_Label', axis=1)
testDF=testDF.set_index(['Class','Label'])   


loadingsDF_met = PCA_analysis_byClass(testDF.fillna(1), 'met_loadings_byClass', colorlabs1, classes1, nloadings=25, mypalette='tab20',export=True, labeled_classes=labeled_classes)
loadingsDF_met.to_csv('met_loadings.csv')

loadingsDF_pY = PCA_analysis_byClass(testDF.fillna(1), 'PRM_pY_loadings_byClass', colorlabs1, classes1, nloadings=25, mypalette='tab20',export=True, labeled_classes=labeled_classes)
loadingsDF_pY.to_csv('PRM_loadings.csv')







#---------------------------------------------------------------------------------------------------------------------------------------------------------

def PCA_x_wise_analysis(df, eT):
    x = df.values  # Use the values from the DataFrame directly

    # Standardize the data
    x_st = StandardScaler().fit_transform(x)

    pca_out = PCA().fit(x_st)
    loadings = pca_out.components_
    num_pc = pca_out.n_features_

    cumsums = np.cumsum(pca_out.explained_variance_ratio_)

    pc_list = ["PC" + str(i) for i in range(1, num_pc + 1)]
    loadings_df = pd.DataFrame(loadings.T, columns=pc_list, index=df.columns)

    pca_scores = pca_out.transform(x_st)

    pca = PCA(n_components=5)
    components = pca.fit_transform(x_st)

    total_var = pca_out.explained_variance_ratio_.sum() * 100
    exp_var_cumul = np.cumsum(pca_out.explained_variance_ratio_)

    num_components = min(5, components.shape[1])
    sns.set_theme(style='ticks')

    # Create a grid of plots for the first 5 PCs
    fig, axes = plt.subplots(num_components, num_components, figsize=(15, 15))

    for i in range(num_components):
        for j in range(num_components):
            if i == j:
                sns.histplot(components[:, i], kde=True, ax=axes[i, j])
                axes[i, j].set_title(f'PC{i + 1}')
            else:
                axes[i, j].scatter(components[:, j], components[:, i], marker='o')
                axes[i, j].set_xlabel(f'PC{j + 1} (' + str(round(pca_out.explained_variance_ratio_[j] * 100, 2)) + '%)')
                axes[i, j].set_ylabel(f'PC{i + 1} (' + str(round(pca_out.explained_variance_ratio_[i] * 100, 2)) + '%)')

    # Adjust layout
    plt.tight_layout()

    # Save and show plot
    plt.savefig(eT + 'x_PCA_grid_plots.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    # Plotting Loadings for the first 5 PCs
    plt.figure(figsize=(15, 10))
    for i in range(num_components):
        plt.plot(df.columns, loadings.T[i], label=f'PC{i + 1}')

    plt.title('Loadings for the First 5 PCs')
    plt.xlabel('Features')
    plt.ylabel('Loadings')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(eT + '_loadings_plot.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    



PCA_x_wise_analysis(pY_PRM_70.fillna(1),'PRM_PCA_x_wise')


