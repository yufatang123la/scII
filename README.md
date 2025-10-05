**scII: Dual-Threshold Adaptive Integration of Single-Cell Multi-omics Data Driven by Imputation**

![Image text]([https://github.com/yufatang123la/scII/blob/21dd07e73dfd5a44206ccf902e230898fb3d407f/fig/abstract_fig.png])
![image](url) 
 
![image](https://github.com/yufatang123la/scII/blob/21dd07e73dfd5a44206ccf902e230898fb3d407f/fig/abstract_fig.png)
 
![image](fig/abstract_fig.png)
**Installation**

scII need to install python first, then pytorch and the necessary library files.

**Preparing intput **

The.h5ad file is the input dataset, the expression matrix for scRNA-seq data are the gene expression matrix (either normalised or raw data), and gene actvitiy matrix for scATAC-seq data.

**Running**

python main.py

**Output**

Output the value of the evaluation metric and the prompt indicating successful integration in the python terminal.

**Visualisation**

scII_batch.tif, .scII_Cluster.tif, .scII_CellTypeProbability.tif are respectively the integrated batch map, cell clustering map, and predicted heat map of cell types in scATAC data.
