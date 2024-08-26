# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:45:11 2021

@author: tytamir

When exporting data from pd, use R friendly txt format, and apply the export layout -- exportlyts (or use TYT search workflows that automatically apply this layout)
Make sure you are in the correct directory 
The script below needs a library or metafile that can extract ncessary data from the csv files. Be sure your csvs have the same column names as your library/metafile.
You can define the experiment type in expType, so that you don not need to edit files that get exported along the way, and you can automatically correct with sup
reachout if you have any questions -- tytamir@mit.edu
"""
import numpy as np
import pandas as pd
import glob
from pyteomics import mgf
import re
from natsort import natsorted
#from datatest import validate

#%matplotlib inline

def PSM_filter(df,libs):
    newdf=pd.DataFrame()
    
	#PSMs with search engine rank of 1
    PSMdf=df[df['Search Engine Rank']==1]
	#select psms with dM ppm between -10 - 10
    PSMdf=PSMdf[PSMdf['Delta M in ppm'].between(-10,10)]
	#select psms with Mascot expectation value less than 0.05
    PSMdf=PSMdf[PSMdf['Expectation Value']<0.05]
    #TMT labeled
    PSMdf=PSMdf[PSMdf['Modifications'].str.contains("TMT")]
	#select psms with ions score of 20 or more
    PSMdf=PSMdf[PSMdf['Ions Score']>=15]
    
    #Select appropriate columns and add gene name column
    for key in libs:
        if key in PSMdf.columns:
            newdf=pd.concat([newdf,PSMdf[key]], axis=1)
            
    
    return newdf

def nan_imputation(df, mgf):
    mgf_df = pd.DataFrame.from_dict(mgf)

    # Expand the params column into separate DataFrame columns containing scan number, charge, monisotopic peptide m/z
    params_df = pd.json_normalize(mgf_df['params'])
    pepmass_df=params_df['pepmass'].astype('string').str.strip().str.replace("\(|'|\)", "", regex=True).str.split(", ", n=1, expand = True)
    params_df = pd.concat([params_df.drop('pepmass', axis=1), pepmass_df[0]], axis=1)
    params_df =params_df.rename(columns={0:'pepmass'})
    # params_df['charge']=params_df['charge'].astype('string').str.strip().str.replace("+", "", regex=True)
    params_df["charge"] = (params_df["charge"].astype("string").str.strip().str.replace("\+", "", regex=True))
    # Concatenate the original DataFrame with the expanded params DataFrame only taking the scan number, charge, and peptide m/z
    mgf_df = pd.concat([mgf_df.drop('params', axis=1), params_df[['scans','charge','pepmass']]], axis=1)

    # Calculate the minimum and maximum intensity values for each scan
    intensity_min_max_df = mgf_df['intensity array'].apply(lambda x: pd.Series({'intensity_min': x.min(), 'intensity_max': x.max()}))

    # Concatenate the original DataFrame with the intensity min/max DataFrame
    mgf_df = pd.concat([mgf_df.drop(['m/z array','intensity array', 'charge array'], axis=1), intensity_min_max_df], axis=1)
    
    
    
    #format mgf dataframe to have mathcing columns as psms dataframe
    mgf_df=mgf_df.rename(columns={'scans':'First Scan', 'charge':'Charge', 'pepmass':'mz in Da'})
    mgf_df['First Scan']=mgf_df['First Scan'].astype('int64')
    mgf_df['Charge']=mgf_df['Charge'].astype('int64')
    mgf_df['mz in Da']=mgf_df['mz in Da'].astype('float64')

    #for psms with missing values match the peptide m/z, charge state, and scan number to the mgf_df and add the minimum intensity
    mdf=pd.merge(df, mgf_df[['First Scan', 'Charge', 'mz in Da','intensity_min', 'intensity_max']], on=['First Scan', 'Charge', 'mz in Da'])

	#drop unused columns
    PSMfilter=mdf.drop(['Search Engine Rank'], axis=1)

	#Count number of channels present, for filtering for data containing all channels used
    PSMfilter=PSMfilter.assign(countTMTchannels=PSMfilter.loc[:,PSMfilter.columns.str.contains("Abundance")].count(axis=1))

    #filter out any psms with missing values in more than 25% of the TMT channels
    PSMfilter=PSMfilter[PSMfilter['countTMTchannels']>=((PSMfilter.columns.str.contains("Abundance")).sum())//4]
    tmtColumns=PSMfilter.columns[PSMfilter.columns.str.contains("Abundance")]
    PSMfilter[tmtColumns.tolist()]=PSMfilter[tmtColumns.tolist()].apply(lambda x: x.fillna(PSMfilter['intensity_min']))
    # abund=PSMfilter.loc[:,PSMfilter.columns.str.contains("Abundance")]
    # PSMfilter=PSMfilter.assign(AbundAve=abund.mean(axis=1))

    return PSMfilter

#this function filters out psms with nan values in tmt channels and low tmt intensity 
def phos_filter(df, expType):	
    #drop unused columns
    phosfilter=df.drop(['Annotated Sequence','Delta M in ppm', 'Ions Score','Delta mz in Da','Expectation Value','First Scan', 'Charge', 'mz in Da','intensity_min', 'intensity_max','countTMTchannels' ], axis=1)
    
    #only keep phospho peptides with ptmRS probablity > 50
    phosfilter=phosfilter[phosfilter['Modifications'].str.contains("Phos")]
    
    #only keep pY peptides, can be commented out if you are working with other types of phospho enrichment
    if expType.endswith('pY'):
        phosfilter=phosfilter[phosfilter['PhosphoRS Best Site Probabilities'].str.contains("Y")]
        
    
    a=pd.DataFrame(phosfilter['PhosphoRS Best Site Probabilities'].str.findall(r"\b[-+]?(?:\d*\.\d+|\d+)").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True).str.split(',', expand=True))
    a=a.replace(r'^\s*$', np.nan, regex=True).astype(float)
    a=a.assign(low=a.min(axis=1))
    phosfilter['ptmRSprobability']=a['low']
    phosfilter=phosfilter[phosfilter['ptmRSprobability']>50]
    phosfilter=phosfilter.drop('ptmRSprobability',axis=1)
    	
    return phosfilter


#function to sum psms, add sites and motifs column
def sum_psms(df):
	#get phosphoRS localization of phosphorylation sites, and parse
    pRS=pd.DataFrame(df['PhosphoRS Best Site Probabilities'].str.findall(r"(\b[S,T,Y]\w+)").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True))

    #concatinate parsed phospohRS column with dataframe    
    phosSum=(pd.concat([df.drop('PhosphoRS Best Site Probabilities',axis=1),pRS], axis=1))
    
    phosSum=phosSum.set_index(['Sequence','PhosphoRS Best Site Probabilities','Master Protein Descriptions','Master Protein Accessions','Modifications'])
    
    #sum peptides based on identifier columns, reset index, capitalize annotated sequence and store in sperate column to match with peptide dataframe
    phosSum=phosSum.groupby(['Master Protein Descriptions','Master Protein Accessions','PhosphoRS Best Site Probabilities']).agg('sum')
    phosSum=phosSum.reset_index()
    
    
    return phosSum


# Define a function to extract the gene names and peptide ranges from the "Master Protein Descriptions" column
def extract_info(s):
    # Split the string by semicolons and whitespace
    items = re.split('; | ', s)
    # Extract the gene names by keeping only the substrings that start with a capital letter
    genes = ','.join([item for item in items if '-' not in item])
    # Extract the peptide ranges by finding the substrings that start with a letter and end with a number
    ranges = ';'.join([item for item in items if '-' in item])
    return genes, ranges

# Define a function to calculate the protSite column value
def calculate_prot_site(peptSite, peptide_range):
    site_list = []
    for site in peptSite.split(','):
        position = int(re.search(r'\d+', site).group())
        aa=str(re.search(r'[A-Z]+', site).group())
        for ranges in peptide_range.split(';'):
            prot_position = position + int(re.search(r'\d+', ranges).group())-1
            if aa+'{}'.format(prot_position) not in site_list:
                site_list.append(aa+'{}'.format(prot_position))
    return ', '.join(site_list)


#Mean center and z-score the data to get fold change over the average value per row (i.e. per phosphosite)
def meanCenter_data(df,eT):
    df=df.assign(ave=df.loc[:,df.columns].mean(axis=1))
    dfmc=df.loc[:,df.columns].div(df['ave'], axis=0)
    dfmc=dfmc.drop(['ave'], axis=1)
    df=df.assign(stndev=df.loc[:,df.columns].std(axis=1))
    dfzs=(df.loc[:,df.columns].sub(df['ave'], axis=0)).div(df['stndev'], axis=0)
    dfzs=dfzs.drop(['ave','stndev'], axis=1)

    return dfmc, dfzs

#this function to be used when bridging runs, may not be needed for most occassions. It normalizes the data to a bridge channel denoted as B#
def bridgeCenter_data(df, brg):
    dfbrg=df.loc[:,df.columns].div(df[df.columns[df.columns.str.contains(pat=brg)]].values,axis=0)
    dfbrg=dfbrg.drop(list(dfbrg.filter(regex=brg)), axis=1)
    #dfbrg.to_csv(eT+'_brg.csv')
   # dfbrglg2=dfbrg.transform(lambda x: np.log2(x))
    #dfbrglg2.to_csv(eT+'_brglg2.csv')
    
    return dfbrg #, dfbrglg2

#import PhosphoSitePlus csv file for motif assignments. Using human proteome for this project. There is also separate mouse and rat PhosphoSitePlus csv files. 
phosSite=pd.read_csv('Phosphosite Motifs_M.csv', sep=',')

dataDF=pd.read_csv('metaFiles_pY.csv',sep=',').set_index('expType')

sumPSM={}
corrDFs={}
libsDict={}
corrs={}
def process_dataset(index):
    #import search results and experimental detail library
    psms = pd.read_csv(dataDF.loc[index,"PSMs"], sep='\t')
    mgf_dict = mgf.read(dataDF.loc[index,"mgf_file"])
    libs = pd.read_csv(dataDF.loc[index,"library"]).set_index('headers').to_dict()['names']
    corrSum = pd.read_csv(dataDF.loc[index,"sup_corr"], sep=',')
    libsDict[index] = libs
    corrs[index]=corrSum
    
    PSMdf=phos_filter(nan_imputation(PSM_filter(psms, libs),mgf_dict),index)
    sumPSMdf=sum_psms(PSMdf)
    
    #for PRM the fasta does not have swissprot formatted Protein Descriptions, so use peptide indexing to get data out
    # Apply the function to the "Master Protein Descriptions" column to create new columns
    genePept = sumPSMdf[['Master Protein Descriptions','PhosphoRS Best Site Probabilities']]
    genePept=genePept.rename(columns={'Master Protein Descriptions':'Gene_peptide', 'PhosphoRS Best Site Probabilities':'peptSite'})
    genePept[['Gene', 'Peptide Ranges']] = genePept['Gene_peptide'].apply(lambda x: pd.Series(extract_info(x)))

    # Add the 'protSite' column to
    genePept['protSite'] = genePept.apply(lambda x: calculate_prot_site(x['peptSite'], x['Peptide Ranges']), axis=1)

    #last set of sum of psms based on genes, UniprotID, Sites, and Motifs
    sumPSMdf=sumPSMdf.rename(columns={'Master Protein Descriptions':'Gene_peptide', 'PhosphoRS Best Site Probabilities':'peptSite'})
    pSumfinal= pd.concat([genePept[['Gene','protSite']], sumPSMdf.drop(['Gene_peptide','peptSite','Master Protein Accessions'], axis=1)], axis=1)
    pSumfinal=pSumfinal.groupby(['Gene','protSite']).agg('sum')
    
    sumPSM[index]=pSumfinal
    
    corrPhos=pSumfinal.copy()
    for col in corrPhos.columns:
        if col in corrSum.columns:
            corrPhos[col]=corrPhos[col]/corrSum[col][0]
    

    #rename generic columns with your sample names
    for x in corrPhos.columns:
        if x in libs.keys():
            corrPhos=corrPhos.rename(columns=libs)
    corrDFs[index]=corrPhos
    corrPhos.to_csv(index+'_corr.csv')

for index in dataDF.index:
    process_dataset(index)

def concat_dfs(data):
    
    # Concatenate the DataFrames along the columns (axis=1)
    df_u = pd.concat(data, axis=1)
    df_u.columns=df_u.columns.droplevel(0)
    df_u=df_u[natsorted(df_u.columns)]
    
    df_n= df_u.dropna()

    return df_u, df_n

# Common function for processing s4G10 and bioxell
def process_runs(keys, prefix, output_prefix):
    run_corr = {}
    wMC = {}
    wMC_zscore = {}
    raw_mc = {}
    raw_zscore = {}
    brg_corr={}
    allBrgs = pd.concat([corrDFs[key] for key in keys], axis=1)
    #allBrgs.columns = allBrgs.columns.droplevel(0)
    allBrgs = allBrgs[natsorted(allBrgs.columns)]
    allBrgs = allBrgs.dropna()

    filtered_cols = allBrgs.filter(regex='A B.*\d+')
    mcB_run, _ = meanCenter_data(filtered_cols, "brg")
    mcB_run = mcB_run.mean(axis=0)
    mcB_run = mcB_run.to_frame().transpose()
    
    for k in keys:
        brgA=bridgeCenter_data(corrDFs[k], 'A B.*\d+')
        brg_corr[k+'_brgA']= brgA

    for k in keys:
        raw_mc[k], raw_zscore[k] = meanCenter_data(corrDFs[k], f'{k}_raw')
        for col in corrDFs[k].columns:
            if col in mcB_run.columns:
                df = corrDFs[k] / mcB_run[col][0]
                df = df.drop([col for col in df.columns if "A B" in col and any(char.isdigit() for char in col)], axis=1)
                run_corr[k] = df
                wMC[k], wMC_zscore[k] = meanCenter_data(df, k)
    
    concatDict={}
    
    concatDict['runCorr_DF_u'], concatDict['runCorr_DF_n'] = concat_dfs(run_corr)
    concatDict['run_corr_mc_u'], concatDict['run_corr_mc_n'] = concat_dfs(wMC)
    concatDict['run_corr_zs_u'], concatDict['run_corr_zs_n']=concat_dfs(wMC_zscore)
    concatDict['raw_mc_DF_u'], concatDict['raw_mc_DF_n']=concat_dfs(raw_mc)
    concatDict['raw_zscore_DF_u'],concatDict['raw_zscore_DF_n'] =concat_dfs(raw_zscore)
    concatDict['brgA_u'],concatDict['brgA_n'] =concat_dfs({key: value for key, value in brg_corr.items() if 'brgA' in key})
    
    return mcB_run, concatDict

# Process s4G10
s4G10={}
s4G10_runFactors, s4G10=process_runs(['PRM1_pY', 'PRM2_pY', 'PRM3_pY', 'PRM4_pY', 'PRM5_pY'], 'PRM', '_pY')

s4G10_runFactors.to_csv("pY_PRM1_runCorrection_factors.csv")
for i, j in s4G10.items():
    j.to_csv("PRM1_pY_"+i+".csv")
    
    
    
    
    



