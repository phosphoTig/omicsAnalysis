# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:45:11 2021

@author: tytamir

When exporting data from pd, use R friendly txt format, and apply the export layout -- exportlyts (or use TYT search workflows that automatically apply this layout)
Make sure you are in the correct directory 
The script below needs a library or metafile that can extract necessary data from the csv files. Be sure your csvs have the same column names as your library/metafile.
You can define the experiment type in expType, so that you do not need to edit files that get exported along the way, and you can automatically correct with sup
reachout if you have any questions -- tytamir@mit.edu
"""
import numpy as np
import pandas as pd
import glob
from pyteomics import mgf
from natsort import natsorted

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
            
    data1 = newdf["Master Protein Descriptions"].str.split("GN=", n=1, expand = True)
    data = data1[1].str.split(" ", n=1, expand = True)
    newdf["Gene"]=data[0]
    
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


#this function filters out psms with high isolation interference, nan values in tmt channels, low tmt intensity 
def phos_filter(df, expType):   
    # drop unused columns
    phosfilter = df.drop(
        [
            "Delta M in ppm",
            "Ions Score",
            "Delta mz in Da",
            "Master Protein Descriptions",
        ],
        axis=1,
    )
    	
    	
    #calcualte average abundance for each peptide
    abund=phosfilter.loc[:,phosfilter.columns.str.contains("Abundance")]
    phosfilter=phosfilter.assign(AbundAve=abund.mean(axis=1))
    phosfilter=phosfilter[phosfilter['AbundAve']>=1000]
    phosfilter=phosfilter.drop('AbundAve', axis=1)
    
    #only keep phospho peptides with ptmRS probablity > 50
    phosfilter=phosfilter[phosfilter['Modifications'].str.contains("Phos")]
    
    #only keep pY peptides, can be commented out if you are working with other types of phospho enrichment
    if 'pY' in expType:
        phosfilter=phosfilter[phosfilter['Modifications'].str.contains("Y")]
        
    # phosfilter=phosfilter.drop(['AbundAve'], axis=1)
    
    a=pd.DataFrame(phosfilter['PhosphoRS Best Site Probabilities'].str.findall(r"\b[-+]?(?:\d*\.\d+|\d+)").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True).str.split(',', expand=True))
    a=a.replace(r'^\s*$', np.nan, regex=True).astype(float)
    a=a.assign(low=a.min(axis=1))
    phosfilter['ptmRSprobability']=a['low']
    phosfilter=phosfilter[phosfilter['ptmRSprobability']>50]
    phosfilter=phosfilter.drop('ptmRSprobability',axis=1)
    
    return phosfilter
	

#function to sum psms, add sites and motifs column
def sum_psms(df, pepts, mods, phosSite, expType):
    
	#get phosphoRS localization of phosphorylation sites, and parse
    pRS=pd.DataFrame(df['PhosphoRS Best Site Probabilities'].str.findall(r"(\b[S,T,Y]\w+)").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True))

    #concatinate parsed phospohRS column with dataframe    
    phosSum=(pd.concat([df.drop(['PhosphoRS Best Site Probabilities', 'Modifications', 'Expectation Value', 'First Scan', 'Charge', 'mz in Da', 'intensity_min','intensity_max', 'countTMTchannels'],axis=1),pRS], axis=1))
    # phosSum=(pd.concat([df.drop(['Search Engine Rank','PhosphoRS Best Site Probabilities', 'Modifications', 'Expectation Value', 'First Scan', 'Charge', 'mz in Da'],axis=1),pRS], axis=1))
    
    #sum peptides based on identifier columns, reset index, capitalize annotated sequence and store in sperate column to match with peptide dataframe
    phosSum=phosSum.groupby(['Annotated Sequence','Sequence','Gene','Master Protein Accessions', 'PhosphoRS Best Site Probabilities']).agg('sum')
    phosSum=phosSum.reset_index()
    phosSum['Annotated Sequence2']=phosSum['Annotated Sequence'].astype('string').str.upper()
    
    #get modification sites from peptide file
    pSeq= pepts[['Annotated Sequence','Modifications in Master Proteins','Positions in Master Proteins']]
    
    #extract mascot site assignments and peptide start/end columns to add to sumed psm dataframe
    pSeq2=pd.DataFrame(pSeq['Modifications in Master Proteins'].str.findall(r"(?<=\[)([^]]+)(?=\])").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True))
    start=pd.DataFrame(pSeq['Positions in Master Proteins'].str.findall(r"(?<=\[)([^]]+)(?=\])").astype('string').str.strip().str.replace("\[|'|\]", "", regex=True))
    start2= start['Positions in Master Proteins'].str.split(',', expand=True)
    start2=start2[0].str.split('-', expand=True)
    pSeq3=pSeq.assign(Sites=pSeq2,peptStart=start2[0], peptEnd=start2[1])
    pSeq3['Annotated Sequence']=pSeq3['Annotated Sequence'].astype('string')
    pSites=pSeq3.set_index('Annotated Sequence').to_dict()['Sites']
    pStart=pSeq3.set_index('Annotated Sequence').to_dict()['peptStart']
    pEnd=pSeq3.set_index('Annotated Sequence').to_dict()['peptEnd']
    phosSum['Sites'] = phosSum['Annotated Sequence2'].map(pSites)
    phosSum['peptStart'] = phosSum['Annotated Sequence2'].map(pStart)
    phosSum['peptEnd'] = phosSum['Annotated Sequence2'].map(pEnd)
    
    #add phosphoRS localized sites back to the dataframe 
    pRSx=pd.DataFrame(phosSum['PhosphoRS Best Site Probabilities'].str.findall(r'(\d+)').astype('string').str.strip().str.replace("\[|'|\]", "", regex=True))
    phosSum['intSites']=pRSx['PhosphoRS Best Site Probabilities'].str.split(',').astype('string').str.strip().str.replace("\[|'| |\]", "", regex=True)
    phosSum['AA+ptmRS']=phosSum['Sequence']+'_'+phosSum['PhosphoRS Best Site Probabilities']
    pRSdict=phosSum.set_index('AA+ptmRS').to_dict()['intSites']
    peptStartdict=phosSum.set_index('AA+ptmRS').to_dict()['peptStart']
    for k, v in pRSdict.items():
        pRSdict[k]=list(v.split(','))
    
    trueSite={}
    for i, j in peptStartdict.items():
        for k, v in pRSdict.items():
            if i == k:
                trueSite[k] = '; '.join([str(int(j) - 1 + int(vi)) for vi in v])
          
    phosSum['trueSite']=phosSum['AA+ptmRS'].map(trueSite)
    uniprotID=phosSum['Master Protein Accessions'].str.split(';', expand=True)
    phosSum['Accessions']=uniprotID[0]
    phosSum=phosSum.drop(['intSites','Annotated Sequence','AA+ptmRS','Annotated Sequence2','Sites','Master Protein Accessions'], axis=1)
    phosSum=phosSum.groupby(['Sequence','Gene','Accessions', 'peptStart', 'peptEnd', 'trueSite','PhosphoRS Best Site Probabilities']).agg('sum')
    
    
    #change column name for peptide sequence to make it easier to map to dictionary containing motifs from the modification file
    phosSum1=phosSum.reset_index().rename(columns={'Sequence':'Peptide Sequence', 'PhosphoRS Best Site Probabilities':'ptmRS'})
    
    #use dictionary map function to add sites from the ptmRS results column
    psiteA=phosSum1.to_dict()['trueSite']
    for k, v in psiteA.items():
        psiteA[k]=list(v.split('; '))
    phosSum1['ptmRS']=phosSum1['ptmRS'].replace('\d+','', regex=True)
    ptmRSdict=phosSum1.to_dict()['ptmRS']
    for k, v in ptmRSdict.items():
        ptmRSdict[k]=list(v.split(', '))
    
    newSite=[]
    for i in range(len(psiteA)):
       newSite.append(', '.join([ptmRSdict[i][j] + psiteA[i][j] for j in range(len(psiteA[i]))])) 
    
    phosSum1=phosSum1.drop('trueSite',axis=1)
    phosSum1['Site']=newSite
    
    
    #find peptides with the same site but with missed cleavage, and make the petptide sequences match, then group peptides again 
    for i in phosSum1['Peptide Sequence']:
        if i.endswith('KK') or i.endswith('RK') or i.endswith('KR'):
            phosSum1['Peptide Sequence']=phosSum1['Peptide Sequence'].replace(i,i[:-1])
    phosSum1['ID']=phosSum1['Accessions']+'_'+phosSum1['Peptide Sequence']
    phosSum1=phosSum1.groupby(['Peptide Sequence','Gene','Accessions','Site','ptmRS','ID']).agg('sum',numeric_only=True).reset_index()
    
    
    #cleanup the modifications table and extract necessary columns for motif extraction
    mods1=mods[mods['Confidence']=='High']
    if 'pY' in expType:
        mods1=mods1[mods1['Target Amino Acid']=='Y']
    mods1=mods1[['Target Amino Acid','Position in Peptide','Peptide Sequence','Protein Accession','Position','Motif']]
    mods1['Site in peptide']=mods1['Target Amino Acid'].astype('string') + mods1['Position in Peptide'].astype('string')
    mods1['Site in Protein']=mods1['Target Amino Acid'].astype('string') + mods1['Position'].astype('string')
    mods1['ID']=mods1['Protein Accession']+'_'+mods1['Peptide Sequence']
    mods2=mods1[['ID','Site in Protein','Motif']]

    #convert modifications dataframe into a dictionary with key=peptide sequence and value=motif, then map your summed psms dataframe to add a motifs column
    modsdict={k: f.groupby('Site in Protein')['Motif'].apply(list).to_dict() for k, f in mods2.groupby('ID')}
    
    #convert PhosphoSitePlus dataframe into a dictionary with key=acceession and value={Site:Motif}
    phosSiteDict={k: f.groupby('Site')['Motif'].apply(list).to_dict() for k, f in phosSite.groupby('Accession')}
    
    #I prefer to work off of copies of dataframes so that I don't need to re-run whole functions again.
    phosSum2=phosSum1.copy()

    
    #get motif direclty from mods dataframe
    for k, v in modsdict.items():
        for i in range(len(phosSum2)):
            if k==phosSum2['ID'][i]:
                m=[]
                for j in list(phosSum2['Site'][i].split(', ')):
                    m.extend(v.get(j, 'X'))
                phosSum2.loc[i,'Motif']=(', '.join(m))
            

    #use PhosphoSitePlus dataframe to assign Sites with motifs
    phosSum3=phosSum2.fillna('X')
    for f, g in phosSiteDict.items():
        for i in range(len(phosSum3)):
            if f == phosSum3['Accessions'][i] and 'X' in phosSum3['Motif'][i]:
                x = []
                for n, motif_part in zip(list(phosSum3['Site'][i].split(', ')), list(phosSum3['Motif'][i].split(', '))):
                    if 'X' in motif_part:
                        x.extend(g.get(n, 'X'))
                    else:
                        x.extend([motif_part])
                phosSum3.loc[i, 'Motif'] = (', '.join(x))
    #if there are still rows without motifs, cut your losses and assign the peptide sequence to it, congrats you have an unseen site, it is a trap!
    phosSum4=phosSum3.copy() 
    for i in range(len(phosSum4)):
        if 'X' in phosSum4['Motif'][i]:
            x = []
            for motif_part in list(phosSum4['Motif'][i].split(', ')):
                if 'X' in motif_part:
                    x.append(phosSum4['Peptide Sequence'][i])
                else:
                    x.append(motif_part)
            phosSum4.loc[i, 'Motif'] = (', '.join(x))
            
    phosSum4=phosSum4.drop(['Peptide Sequence','ptmRS','ID'], axis=1)
    phosSum4=phosSum4.groupby(['Gene','Accessions', 'Site', 'Motif']).agg('sum', numeric_only=True)
    phosSum4.to_csv(expType+'sum_motif.csv')
    phosSum5=phosSum4.reset_index().drop('Motif', axis=1).set_index(['Gene','Accessions', 'Site']).groupby(['Gene','Accessions', 'Site']).agg('sum', numeric_only=True)
    return phosSum5

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

dataDF=pd.read_csv('metaFiles.csv',sep=',').set_index('expType')

sumPSM={}
corrDFs={}
def process_dataset(index):
    #import search results and experimental detail library
    psms = pd.read_csv(dataDF.loc[index,"PSMs"], sep='\t')
    pepts=pd.read_csv(dataDF.loc[index,"pepts"], sep='\t')
    mods=pd.read_csv(dataDF.loc[index,"mods"], sep='\t')
    mgf_dict = mgf.read(dataDF.loc[index,"mgf_file"])
    libs = pd.read_csv(dataDF.loc[index,"library"]).set_index('headers').to_dict()['names']
    corrSum = pd.read_csv(dataDF.loc[index,"sup_corr"], sep='\t')
    
    PSMdf=phos_filter(nan_imputation(PSM_filter(psms, libs),mgf_dict),index)
    sumPSMdf=sum_psms(PSMdf, pepts, mods, phosSite, index)
    sumPSM[index]=sumPSMdf
    
    corrPhos=sumPSMdf.copy()
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
        brg_corr[k+'_brgS']= bridgeCenter_data(brgA, 'B.*\d+')

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
    concatDict['brgS_u'],concatDict['brgS_n'] =concat_dfs({key: value for key, value in brg_corr.items() if 'brgS' in key})
    
    return mcB_run, concatDict

# Process s4G10
s4G10={}
s4G10_runFactors, s4G10=process_runs(['F1_pY_a_DDA', 'F2_pY_a_DDA', 'F3_pY_a_DDA', 'M1_pY_b_DDA', 'M2_pY_a_DDA', 'M3_pY_a_DDA'], 's4G10', 's4g10_pY')

s4G10_runFactors.to_csv("pY_s4G10_runCorrection_factors.csv")
for i, j in s4G10.items():
    j.to_csv("s4G10_pY_"+i+".csv")

# Process bioxell
bioxell={}
bioxell_runFactors, bioxell=process_runs(['F1b_pY_DDA', 'F2b_pY_DDA', 'F3b_pY_DDA', 'M1c_pY_DDA', 'M2b_pY_DDA', 'M3b_pY_DDA'], 'bioxell', 'bioxell_pY')

bioxell_runFactors.to_csv("pY_bioxell_runCorrection_factors.csv")
for i, j in bioxell.items():
    j.to_csv("bioxell_pY_"+i+".csv")








