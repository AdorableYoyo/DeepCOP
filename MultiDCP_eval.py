import pandas as pd
import numpy as np 
import numpy as np
from pathlib import Path
from Helpers.data_loader import get_feature_dict, load_csv
from Helpers.utilities import all_stats
from tensorflow.keras.models import model_from_json
# load model

drugs_phase2 = pd.read_csv('/raid/home/yoyowu/DeepCOP/Data/phase2_compounds_morgan_2048.csv',index_col=0)

testdata= pd.read_csv('/raid/home/yoyowu/DeepCOP/deepCOP_data_filtered.csv',index_col=0)

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

testdata = testdata.rename(columns={
                        'TOMM70A' : 'TOMM70',
                        'KIAA0196' : 'WASHC5',
                        'KIAA1033' : 'WASHC4',
                        'PRUNE' : 'PRUNE1',
                        'ADCK3' : 'COQ8A',
                        'LRRC16A' : 'CARMIL1',
                        'FAM63A' : 'MINDY1'})

### load the model 

def load_model(file_prefix):
    # load json and create model
    json_file = open(file_prefix + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_prefix + '.h5')
    print("Loaded model", file_prefix)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)


  # load the models
up_model = load_model_from_file_prefix('/raid/home/yoyowu/DeepCOP/SavedModels/DCPcells_Up_5p_27')
down_model = load_model_from_file_prefix('/raid/home/yoyowu/DeepCOP/SavedModels/DCPcells_Down_5p_27')

# build your input features
gene_features_dict = get_feature_dict("Data/go_fingerprints.csv")
drug_features_dict = get_feature_dict("DCP_morgan_2048.csv")
#drug_features_dict.pop("Enzalutamide")
#drug_features_dict.pop("VPC14449")
# drug_features_dict.pop("VPC17005")

data = []
descriptions = []


for drug in drug_features_dict:
    for gene in gene_features_dict:
        data.append(drug_features_dict[drug] + gene_features_dict[gene])
        descriptions.append(drug + ", " + gene)
data = np.asarray(data, dtype=np.float16)

# get predictions
up_predictions = up_model.predict(data)
down_predictions = down_model.predict(data)



down_lables = pd.read_csv('filtered_down_lables.csv',index_col=0)
up_lables = pd.read_csv('filtered_up_lables.csv',index_col=0)
dict_down_lables = down_lables.to_dict('dict')
dict_up_lables = up_lables.to_dict('dict')
up_Y= []
down_Y =[]
descriptions_Y =[]

for drug in drug_features_dict:
    for gene in gene_features_dict:
        y_down = dict_down_lables[gene][drug]
        y_up = dict_up_lables[gene][drug]
        down_Y.append(y_down)
        up_Y.append(y_up)
        descriptions_Y.append(drug + ", " + gene)
down_Y = np.asarray(down_Y, dtype=np.float16)
up_Y = np.asarray(up_Y, dtype=np.float16)

assert(len(down_Y)==len(down_predictions))


down_stats = all_stats(np.asarray(down_Y, dtype='float32'), down_predictions[:, 1])


print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff | Max F Score')
print('All stats down:', ['{:6.3f}'.format(val) for val in down_stats])

down_auc = float(down_stats[0])
down_maxfscore = float(down_stats[6])

up_stats = all_stats(np.asarray(up_Y, dtype='float32'), up_predictions[:, 1])


print('All stats columns | AUC | Recall | Specificity | Number of Samples | Precision | Max F Cutoff | Max F Score')
print('All stats Up:', ['{:6.3f}'.format(val) for val in up_stats])

up_auc = float(down_stats[0])
up_maxfscore = float(down_stats[6])
print(f'auc and fscore of down gene is {down_auc}{down_maxfscore}, up gene {up_auc}{up_maxfscore}')