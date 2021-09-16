import pandas as pd
import numpy as np 
testdata=pd.read_csv('/raid/home/yoyowu/DeepCOP/deepCOP_data_filtered.csv',index_col=0)
totaldata=pd.read_csv('signature_total.csv',index_col=0)

totaldata=totaldata.drop(['cell_id','pert_idose','pert_type'],axis=1)
totaldata=totaldata.set_index(['pert_id'])

gene_cutoffs_down={}
gene_cutoffs_up={}
percentile_down = 5
percentile_up = 100-percentile_down

for gene in totaldata.columns:
     row = totaldata[str(gene)]
     gene_cutoffs_down[gene] = np.percentile(row, percentile_down)
     gene_cutoffs_up[gene] = np.percentile(row, percentile_up)

all_genes = list(totaldata.columns)
new_test_down = pd.DataFrame()
new_test_up = pd.DataFrame()
for gene in all_genes:    
      new_test_down[gene] = testdata[gene]<gene_cutoffs_down[gene]
      new_test_up[gene] = testdata[gene]>gene_cutoffs_up[gene]  

new_test_down = new_test_down.astype(int)
new_test_up = new_test_up.astype(int)
assert(len(new_test_down)==503)
assert(len(new_test_up)==503)
ratio_pos_down =  (np.count_nonzero(new_test_down))/(503*978)
ratio_pos_up =  (np.count_nonzero(new_test_up))/(503*978)

print(f'the ratio of positive down regulated genes and up regulated genes are {ratio_pos_down} , {ratio_pos_up}')



new_test_up = new_test_up.rename(columns={
                        'TOMM70A' : 'TOMM70',
                        'KIAA0196' : 'WASHC5',
                        'KIAA1033' : 'WASHC4',
                        'PRUNE' : 'PRUNE1',
                        'ADCK3' : 'COQ8A',
                        'LRRC16A' : 'CARMIL1',
                        'FAM63A' : 'MINDY1'})

new_test_down = new_test_down.rename(columns={
                        'TOMM70A' : 'TOMM70',
                        'KIAA0196' : 'WASHC5',
                        'KIAA1033' : 'WASHC4',
                        'PRUNE' : 'PRUNE1',
                        'ADCK3' : 'COQ8A',
                        'LRRC16A' : 'CARMIL1',
                        'FAM63A' : 'MINDY1'})

new_test_up.to_csv('filtered_up_lables.csv')
new_test_down.to_csv('filtered_down_lables.csv')
