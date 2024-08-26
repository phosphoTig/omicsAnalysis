import numpy as np
import pandas as pd
import glob
from pyteomics import mgf
import re
from natsort import natsorted


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

    #filter out any psms with missing values in more than a third of the TMT channels
    PSMfilter=PSMfilter[PSMfilter['countTMTchannels']>=((PSMfilter.columns.str.contains("Abundance")).sum())//3]
    tmtColumns=PSMfilter.columns[PSMfilter.columns.str.contains("Abundance")]
    PSMfilter[tmtColumns.tolist()]=PSMfilter[tmtColumns.tolist()].apply(lambda x: x.fillna(PSMfilter['intensity_min']))
    abund=PSMfilter.loc[:,PSMfilter.columns.str.contains("Abundance")]
    PSMfilter=PSMfilter.assign(AbundAve=abund.mean(axis=1))

    return PSMfilter

def sup_Corrections(df):
    rawSup=df[df['AbundAve']>=df['AbundAve'].quantile(0.75)]
    rawSup=rawSup.drop(['Expectation Value','First Scan', 'Charge', 'mz in Da','Annotated Sequence','intensity_min','intensity_max','AbundAve', 'countTMTchannels', 'PhosphoRS Best Site Probabilities', 'Modifications', 'Delta M in ppm',	'Delta mz in Da',	'Ions Score',	'AbundAve'], axis=1)
    psmsSum=df.set_index(['Gene','Master Protein Accessions','Sequence'])
    psmsSum=psmsSum.groupby(['Gene','Master Protein Accessions','Sequence']).agg('sum')
    abund=psmsSum.loc[:,psmsSum.columns.str.contains("Abundance")]
    psmsSum2=psmsSum.assign(AbundAve=abund.mean(axis=1))
    psmsSum2=psmsSum.reset_index()
    
    Sup=psmsSum2.loc[:,psmsSum2.columns.str.contains("Abundance")].div(psmsSum2['AbundAve'], axis=0)
    Sup=Sup.mean(axis=0).to_frame().transpose()
    
    pepSum=psmsSum.reset_index()
    uniprotID=pepSum['Master Protein Accessions'].str.split(';', expand=True)
    pepSum['Accessions']=uniprotID[0]
    pepSum['Label']=pepSum['Gene']+'_'+pepSum['Accessions']+'_'+pepSum['Sequence']
    pepSum=pepSum.set_index('Label')
    pepSum=pepSum.drop(['Gene','Accessions','Master Protein Descriptions','Master Protein Accessions','Sequence','Expectation Value','First Scan', 'Charge', 'mz in Da','intensity_min','intensity_max', 'countTMTchannels', 'Annotated Sequence','Sequence','PhosphoRS Best Site Probabilities', 'Modifications', 'Delta M in ppm',	'Delta mz in Da',	'Ions Score',	'AbundAve'], axis=1)
    
    return pepSum, Sup


#function to sum psms, add sites and motifs column
def sum_peps(df):
    df=df.drop(['Expectation Value','First Scan', 'Charge', 'mz in Da','intensity_min','intensity_max', 'countTMTchannels', 'Annotated Sequence','Sequence','PhosphoRS Best Site Probabilities', 'Modifications', 'Delta M in ppm',	'Delta mz in Da',	'Ions Score',	'AbundAve'], axis=1)
    
    uniprotID=df['Master Protein Accessions'].str.split(';', expand=True)
    df['Accessions']=uniprotID[0]
    pepSum=df.set_index(['Gene','Accessions']).drop(['Master Protein Descriptions','Master Protein Accessions'], axis=1)
    
    
    #sum peptides based on gene name and protein accession columns
    pepSum=pepSum.groupby(['Gene','Accessions']).agg('sum')
    
    return pepSum


#Mean center and z-score the data to get fold change over the average value per row (i.e. per phosphosite)
def meanCenter_data(df,eT):
    df=df.assign(ave=df.loc[:,df.columns].mean(axis=1))
    dfmc=df.loc[:,df.columns].div(df['ave'], axis=0)
    dfmc=dfmc.drop(['ave'], axis=1)
    # dfmc.to_csv(eT+'_mc.csv')
    df=df.assign(stndev=df.loc[:,df.columns].std(axis=1))
    dfzs=(df.loc[:,df.columns].sub(df['ave'], axis=0)).div(df['stndev'], axis=0)
    dfzs=dfzs.drop(['ave','stndev'], axis=1)
    # dfzs.to_csv(eT+'_zs.csv')

    return dfmc, dfzs


#this function to be used when bridging runs, may not be needed for most occassions. It normalizes the data to a bridge channel denoted as B#
def bridgeCenter_data(df, brg):
    dfbrg=df.loc[:,df.columns].div(df[df.columns[df.columns.str.contains(pat=brg)]].values,axis=0)
    dfbrg=dfbrg.drop(list(dfbrg.filter(regex=brg)), axis=1)
    #dfbrg.to_csv(eT+'_brg.csv')
   # dfbrglg2=dfbrg.transform(lambda x: np.log2(x))
    #dfbrglg2.to_csv(eT+'_brglg2.csv')
    
    return dfbrg #, dfbrglg2


dataDF=pd.read_csv('metaFiles_PRM_sup.csv',sep=',').set_index('expType')

corrDict={}
corrPeps={}
sumPSM={}
corrDFs={}
libsDict={}
def process_dataset(index):
    psms = pd.read_csv(dataDF.loc[index, "PSMs"], sep='\t')
    mgf_dict = mgf.read(dataDF.loc[index, "mgf_file"])
    libs = pd.read_csv(dataDF.loc[index, "library"]).set_index('headers').to_dict()['names']
    libsDict[index] = libs

    PSMdf = nan_imputation(PSM_filter(psms, libs), mgf_dict)
    peps, corrSum = sup_Corrections(PSMdf)
    corrDict[index] = corrSum
    sumPSMdf = sum_peps(PSMdf)
    sumPSM[index] = sumPSMdf

    corrSum.to_csv(index.replace("_sup",'')+'_sup_Corrections.csv')

    corrProt = sumPSMdf.copy()
    corrPept=peps.copy()
    for col in corrProt.columns:
        if col in corrSum.columns:
            corrProt[col] = corrProt[col] / corrSum[col][0]
    
    for col in corrPept.columns:
        if col in corrSum.columns:
            corrPept[col] = corrPept[col] / corrSum[col][0]

    for x in corrProt.columns:
        if x in libs.keys():
            corrProt = corrProt.rename(columns=libs)
            
    for y in corrPept.columns:
        if y in libs.keys():
            corrPept = corrPept.rename(columns=libs)
            
    
    corrDFs[index] = corrProt
    corrPeps[index] = corrPept
    corrProt.to_csv(index.replace("_sup",'')+'_gProt_corr.csv')
    corrPept.to_csv(index.replace("_sup",'')+'_pepts_corr.csv')

# Process datasets in parallel
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
gProt={}
gProt_runFactors, gProt=process_runs(['PRM1_sup', 'PRM2_sup', 'PRM3_sup', 'PRM4_sup', 'PRM5_sup'], 'PRM1', '_gProt')

gProt_runFactors, gProt=process_runs(['BHA_HFD15_PRM_sup', 'BHA_HFD16_PRM_sup'], 'BHA_PRM', '_gProt')

gProt_runFactors.to_csv("PRM_sup_runCorrection_factors.csv")
for i, j in gProt.items():
    j.to_csv("gProt"+i+".csv")


