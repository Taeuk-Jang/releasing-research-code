## NIPS2020 Learning Fair Representation via Posterior Regularization

#### 1. Specification of dependencies
Specific dependencies about the project can be found in `requirements.txt`

#### 2. code description
to run example training process:
```console
python train.py --dataset 'bank' --senstive_feature 1 --upsample True  --lr_h 1e-3 --lr_c 1e-5 --lr_p 1e-3 --et 1 --lamda 1 --alpha 0.2 --mu 0.6
```

`src/train.py` : it has the process of training FSNS network. It trains for 5 repetition.

`src/test.py` :  it has the process of testing FSNS network. It tests for 5 repetition.

`src/utils.py` : it contains utils to run the model including loss functions and minor calculations.

`src/eval_utils.py` : it contains utilities to visualize how the learned representation is fair or perform well.
`vis(latent, a, y)` visualize the features w.r.t sensitive attribute and target label to show the separation of the learned representation w.r.t. the features. 
`evaluate(args, repeat, epoch, dataset, dataloader, H, P, C, sens_idx, num_sens, privileged_groups, unprivileged_groups,  device, test)` quantifies the performance and fairness.

`src/models.py` : it contains the structures of FSNS modules.


#### 3. Pre-trained models
Example pre-trained model trained on Bank dataset is included in `model/model.ckpt`
The dataset can be downloaded from the link.


If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at jang141@purdue.com or open an issue on this GitHub repository. 
