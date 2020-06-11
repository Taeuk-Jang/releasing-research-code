## NIPS2020 Constructing a Fair Classiﬁer with the Generated Fair Data

#### 1. Specification of dependencies
Specific dependencies about the project can be found in `requirements.txt`

#### 2. code description

`src/train-vae.py` : it has the process of training VAE-GAN network. 
to run example training process on COMPAS dataset:
```console
python train-vae.py --dataset 'compas' --senstive_feature 1 --lr_vae 1e-4 --lr_d 1e-4 --epochs 2000 --lambda_cls 1 --lambda_gp 10 --lambda_rec 10 --lambda_kld 0.1 --n_critic 6
```

`src/train-cls.py` : it has the process of training classifier. 
to run example training process on COMPAS dataset:
```console
python train-cls.py --dataset 'compas' --senstive_feature 1 --lr_c 1e-4 
```

`src/test.py` :  it has the process of testing transfer learning network. It tests for 5 repetition.
you can also try with :
to run example training process:
```console
python test.py --dataset 'compas'
```
`src/utils.py` : it contains utils to run the model including loss functions and minor calculations.

`src/eval_utils.py` : it contains utilities to visualize how the learned representation is fair or perform well.

`src/models.py` : it contains the structures of VAE-GAN modules.


#### 3. Pre-trained models
Example pre-trained model trained on Compas dataset is included in `model/compas/vae.pth`(VAE-GAN), `model/bank/c.pth`(classifier).

The dataset can be downloaded from the [link](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv).
Data pre-processing is done with the package provided by [AIF360](https://github.com/IBM/AIF360)

