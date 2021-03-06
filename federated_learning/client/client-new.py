from opacus.utils import module_modification
from fastai.vision.all import *
from dsail.federated_learning import *
from dsail.utils import *
import os

# / ! \
# This file was created to comply to the Dockerfile
# It is not up to date with the last modifications made to the clients H_federated and H_nofederated


@call_parse
def main(
    arch: Param("Architecture to use", str)='resnet18',
    lr: Param("Learning Rate", int)=3e-3/5,
    epochs: Param("number of epochs for training", int)=1,
    bs: Param("Batch size to use", int)=16,
    device: Param("Which device to use", str)='cuda:0',
    port: Param("The port used for federated learning", int)=8080,
    apply_dp: Param("Learning rate", store_true)=True,
    alphas: Param("Alphas", range)=range(2,32),
    noise_multiplier: Param("Noise injected in DP", int)=0.5, 
    max_grad_norm: Param("Maximum Gradient Norm when clipping", int)=1.0,
    delta: Param("Delta", int)=1e-5,
    matrix_path: Param("Pass a value to save the  confusion matrix", str)=None,
    csv_path: Param("Pass a value to store the logs in csv", str)=None,
    roc_path: Param("Pass a value to store the ROC-AUC curve", str)=None,
    data_path: Param("datapath to use", str)="/dataset",
    seed: Param("Pass a value to set seed", int)=42,

): 
    device=torch.device(device)

    model = globals()[arch]
    hospital_name =  str(os.environ['HOSPITAL_NAME'])

    data_path = str(Path(data_path)) + "/" + str(os.environ['HOSPITAL_NAME'])

    ip_server =  str(os.environ['IP_SERVER'])



#
    print("")
    print("DSAIL | Client Starting")
    print("Dataset: ", data_path)
    print("")
    
    #print(data_path)
    #device=torch.device(device)
    #print(data_path)
    #time.sleep(3000)
    #print(get_image_files,parent_label,GrandparentSplitter())
    #data_path = Path(data_path)
    #dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),get_items = get_image_files,get_y = parent_label,splitter = GrandparentSplitter())
    #dblock.summary(data_path)

#




    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
        get_items = get_image_files,
        get_y = parent_label,
        splitter = GrandparentSplitter())

    
    # print(dblock.datasets(data_path).train.__class__)



    dls = dblock.dataloaders(data_path, bs=bs, num_workers=0)
    #dls = dblock.dataloaders(data_path, bs=bs, num_workers=0, sample=ImbalancedDatasetSampler(dblock.datasets(data_path).train))
    dblock.summary(data_path)



    if seed is not None: set_seed(dls, seed)

    learn = cnn_learner(dls, model, metrics=[accuracy, RocAucBinary()])
    learn.model = module_modification.convert_batchnorm_modules(learn.model)
    
    client = FLClient(learn,lr, epochs, apply_dp, alphas, noise_multiplier, max_grad_norm, delta, device, csv_path, data_path, matrix_path, roc_path)
    fl.client.start_numpy_client(str(ip_server)+ ":" +str(port), client=client)

