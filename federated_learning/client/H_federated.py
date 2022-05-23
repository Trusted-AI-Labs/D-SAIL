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
    apply_dp:    Param("Apply differential privacy", bool_arg)=False,
    alphas: Param("Alphas", range)=range(2,32),
    noise_multiplier: Param("Noise injected in DP", int)=0.5, 
    max_grad_norm: Param("Maximum Gradient Norm when clipping", int)=1.0,
    delta: Param("Delta", int)=1e-5,
    seed: Param("Pass a value to set seed", int)=42,
    split: Param("Pass the split proportion to use", str)='50_33_17',
    db_loc: Param("Pass the path where the database is stored", str)='Hospitals',
    db: Param("Pass the used database", str)='cancer_database',
    res_loc: Param("Pass the path where the results will be stored", str)='results',
    hospital: Param("Pass the hospital number", str)='H0',
    resize: Param("Pass the size of the image (square resize)", int)=50
): 

    device=torch.device(device)

    model = globals()[arch]

    data_path = f'{db_loc}/{db}/Split_{split}/{hospital}'
    results_path = f'{res_loc}/{db}/federated/Split_{split}/{hospital}/'
    matrix_path = os.path.join(results_path, f'epochs{epochs}_{hospital}_confusion_matrix')
    csv_path = os.path.join(results_path, f'epochs{epochs}_{hospital}_metrics')
    roc_path = os.path.join(results_path, f'epochs{epochs}_{hospital}_roc_auc')
    os.makedirs(results_path, exist_ok=True)
    data_path = Path(data_path)

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
        get_items = get_image_files,
        get_y = parent_label,
        item_tfms = [Resize(resize)],
        splitter = GrandparentSplitter(), 
        )


    ds = dblock.datasets(data_path)
    dls = dblock.dataloaders(data_path, bs=bs, device=device, dl_type=WeightedDL, wgts=get_imbalance_weights(ds), num_workers=0)

    if seed is not None: set_seed(dls, seed)

    learn = cnn_learner(dls, model, metrics=[accuracy, RocAucBinary()])
    learn.model = module_modification.convert_batchnorm_modules(learn.model)
    
    client = FLClient(learn,lr, epochs, apply_dp, alphas, noise_multiplier, max_grad_norm, delta, device, csv_path, data_path, matrix_path, roc_path)
    fl.client.start_numpy_client("127.0.0.1:" + str(port), client=client)

