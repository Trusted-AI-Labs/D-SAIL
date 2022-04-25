from opacus.utils import module_modification
from fastai.vision.all import *
from fastai.callback.all import WeightedDL
from dsail.federated_learning import *
from dsail.utils import *
import os

@call_parse
def main(
    arch:  Param("Architecture to use", str)='resnet18',
    lr: Param("Learning Rate", int)=3e-3/5,
    epochs: Param("number of epochs for training", int)=2,
    bs: Param("Batch size to use", int)=64,
    device:  Param("Which device to use", str)='cuda:0',
    port: Param("The port used for federated learning", int)=8080,
    apply_dp:    Param("Apply differential privacy", store_true)=True,
    alphas: Param("Alphas", range)=range(2,32),
    noise_multiplier: Param("Noise injected in DP", int)=0.5, 
    max_grad_norm: Param("Maximum Gradient Norm when clipping", int)=1.0,
    delta: Param("Delta", int)=1e-5,
    seed: Param("Pass a value to set seed", int)=42,
    split: Param("Which data repartition. Or 'H' for the big hospital", str)='50_33_17',
    db_loc: Param("Pass the path where the database is stored", str)='Hospitals',
    db: Param("Pass the used database", str)='cancer_database',
    res_loc: Param("Pass the path where the results will be stored", str)='results',
    hospital: Param("Pass the hospital number", str)="H0"

): 

    print('Binary classifier')
    if split == 'H':
      data_path = f'{db_loc}/{db}/H/'
      hospital = 'H'
    else:
      data_path = f'{db_loc}/{db}/Split_{split}/{hospital}/'
    results_path = f'{res_loc}/{db}/no_federated/Split_{split}/{hospital}/'
    data_path = Path(data_path)
    os.makedirs(results_path, exist_ok=True)

    device=torch.device(device)

    model = globals()[arch]

    data_path = Path(data_path) 
    

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
        get_items = get_image_files,
        get_y = parent_label,
        item_tfms = [Resize(32)],
        splitter = GrandparentSplitter())


    ds = dblock.datasets(data_path)
    dls = dblock.dataloaders(data_path, bs=bs, device=device, dl_type=WeightedDL, wgts=get_imbalance_weights(ds), num_workers=0)

    if seed is not None: set_seed(dls, seed)


    cbs = []
    if apply_dp: cbs.append(DPCallback(alphas=alphas, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,delta = delta, device=device))
    if results_path is not None: cbs.append(CSVLogger(fname=results_path + f'epochs{epochs}_{hospital}_metrics.csv', append=True))
    
    print('Model set up...')
    learn = cnn_learner(dls, model, metrics=[accuracy, RocAucBinary()])
    learn.model = module_modification.convert_batchnorm_modules(learn.model)
    print('Done!')

    n_round = 3
    for _ in range(n_round):
        learn.fine_tune(epochs, lr, cbs=cbs)

    os.makedirs(f'{res_loc}/{db}/no_federated/Split_{split}/weights/', exist_ok=True)
    learn.save(f'{res_loc}/{db}/no_federated/Split_{split}/weights/{db}_{split}_{hospital}')
    
    
    #client = FLClient(learn,lr, epochs, apply_dp, alphas, noise_multiplier, max_grad_norm, delta, device, csv_path, data_path, matrix_path, roc_path)
    #fl.client.start_numpy_client("127.0.0.1:"+str(port), client=client)

