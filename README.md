# omicsAnalysis
Associated with publication Tamir et al 2025 (PMID: 40441152): https://www.sciencedirect.com/science/article/pii/S1097276525004125?via%3Dihub

Pipeline to process and visualize searched phosphoproteomics and metabolomics datasets
- These set of python scripts process peptide spectral matches (PSM) files proteomics/phosphoproteomics LC-MS runs
- example metadata file formats are included for both phosphoproteomics and crude protein runs
- plotting script is an all in one analysis pipline that generates: fold change & FDR analysis, heatmaps, PCA plots, KDE plots, volcano plots, violin plots, and PCA loadings plots
- PLSR analysis pipeline takes metabololite (Y-matrix) and performs regression against phosphosites (X-matrix) using leave-one-out or k-fold cross validation.
