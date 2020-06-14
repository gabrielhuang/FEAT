# Are Few-Shot Learning Benchmarks too Simple ? Solving them without Test-Time Labels

This is the code repository for the experiments of the paper "Are Few-Shot Learning Benchmarks too Simple ? Solving them without Test-Time Labels". 

Thanks to the authors of the FEAT paper for releasing their code, on which we have based our experiments. The original README is below.

For all experiments, we run `python train_fsl.py` with different arguments.
During or after the few-shot learning finetuning, run the summarization script on the appropriate `eval.jl` file.
For instance,
```bash
python summarize.py checkpoints/CUB-ProtoNet-ConvNet-05w05s15q-Pre-DIS/20_0.5_lr0.0001mul10_step_T132.0T264.0_b0.1_bsz100-NoAug/eval.jl
```
- For validation scores, First/Second/Third column are Lowest/Highest/Average validation score. 
- For Test scores, First/Second column are test scores reported from validation epoch with Lowest/Highest validation score.
Third column is average test score over epochs.

## Same-Domain Experiments


### CUB - Conv-4
```bash
 python train_fsl.py --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset CUB --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --init_weights ./saves/initialization/cub/con-pre.pth --eval_interval 1 --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.03,0.1,0.3,1,3,10,30"
```

Output
<pre>
valid_SupervisedAcc                                0.7335+0.0071 (ep0)   0.8046+0.0061 (ep49)  Mean    0.7893
test_SupervisedAcc                                 0.6791+0.0072 (ep0)   <b>0.7533+0.0071</b> (ep49)  Mean    0.7448
valid_UnsupervisedAcc_softmax_reg1                 0.6100+0.0103 (ep0)   0.7256+0.0094 (ep62)  Mean    0.7028
test_UnsupervisedAcc_softmax_reg1                  0.5372+0.0106 (ep0)   <b>0.6613+0.0108</b> (ep62)  Mean    0.6398
valid_ClusteringAcc_sinkhorn_reg1                  0.7053+0.0095 (ep0)   0.7946+0.0094 (ep46)  Mean    0.7760
test_ClusteringAcc_sinkhorn_reg1                   0.6433+0.0093 (ep0)   0.7362+0.0100 (ep46)  Mean    0.7248
valid_ProtoTransductiveProbAcc_reg1                0.7529+0.0072 (ep0)   0.8236+0.0061 (ep49)  Mean    0.8094
test_ProtoTransductiveProbAcc_reg1                 0.6983+0.0073 (ep0)   0.7735+0.0069 (ep49)  Mean    0.7654
valid_ProtoTransductiveDstAcc_reg1                 0.7529+0.0072 (ep0)   0.8236+0.0061 (ep49)  Mean    0.8094
test_ProtoTransductiveDstAcc_reg1                  0.6983+0.0073 (ep0)   0.7735+0.0069 (ep49)  Mean    0.7654
</pre>

### miniImageNet - Conv-4
```bash
python train_fsl.py  --max_epoch 20 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1 --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.03,0.1,0.3,1,3,10,30"
```

Output
<pre>
valid_SupervisedAcc                                0.6660+0.0070 (ep2)   0.6828+0.0066 (ep5)   Mean    0.6768 
test_SupervisedAcc                                 0.7036+0.0066 (ep2)   <b>0.7072+0.0066</b> (ep5)   Mean    0.7025
valid_UnsupervisedAcc_softmax_reg3                 0.5244+0.0092 (ep0)   0.5486+0.0094 (ep8)   Mean    0.5377 
test_UnsupervisedAcc_softmax_reg3                  0.5696+0.0093 (ep0)   <b>0.5757+0.0094</b> (ep8)   Mean    0.5768
valid_ClusteringAcc_softmax_reg3                   0.6314+0.0086 (ep0)   0.6314+0.0086 (ep0)   Mean    0.6314 
test_ClusteringAcc_softmax_reg3                    0.5887+0.0082 (ep0)   0.5887+0.0082 (ep0)   Mean    0.5887
valid_ProtoTransductiveProbAcc_reg3                0.6810+0.0071 (ep2)   0.6988+0.0069 (ep7)   Mean    0.6926
test_ProtoTransductiveProbAcc_reg3                 0.7176+0.0067 (ep2)   0.7176+0.0067 (ep7)   Mean    0.7186
valid_ProtoTransductiveDstAcc_reg3                 0.6810+0.0071 (ep2)   0.6988+0.0069 (ep7)   Mean    0.6926 
test_ProtoTransductiveDstAcc_reg3                  0.7176+0.0067 (ep2)   0.7176+0.0067 (ep7)   Mean    0.7186
</pre>


### miniImageNet - ResNet-12

```bash
python train_fsl.py  --max_epoch 20 --model_class ProtoNet  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.03,0.1,0.3,1,3,10,30"
```

Output
<pre>
valid_SupervisedAcc                                0.8155+0.0058 (ep7)   0.8292+0.0053 (ep2)   Mean    0.8227 
test_SupervisedAcc                                 0.7910+0.0061 (ep7)   <b>0.8040+0.0057</b> (ep2)   Mean    0.7966
valid_UnsupervisedAcc_softmax_reg10                0.7302+0.0096 (ep9)   0.7490+0.0094 (ep2)   Mean    0.7405 
test_UnsupervisedAcc_softmax_reg10                 0.6787+0.0088 (ep9)   <b>0.6986+0.0094</b> (ep2)   Mean    0.6928
valid_ClusteringAcc_softmax_reg10                  0.7959+0.0095 (ep7)   0.8108+0.0095 (ep0)   Mean    0.8023 
test_ClusteringAcc_softmax_reg10                   0.7600+0.0091 (ep7)   0.7733+0.0092 (ep0)   Mean    0.7673
valid_ProtoTransductiveProbAcc_reg10               0.8373+0.0057 (ep7)   0.8542+0.0051 (ep2)   Mean    0.8464
test_ProtoTransductiveProbAcc_reg10                0.8130+0.0061 (ep7)   0.8279+0.0057 (ep2)   Mean    0.8198
valid_ProtoTransductiveDstAcc_reg10                0.8373+0.0057 (ep7)   0.8542+0.0051 (ep2)   Mean    0.8464
test_ProtoTransductiveDstAcc_reg10                 0.8130+0.0061 (ep7)   0.8279+0.0057 (ep2)   Mean    0.8198
</pre>

### tieredImageNet - ResNet-12

```bash
python train_fsl.py  --max_epoch 20 --model_class ProtoNet  --backbone_class Res12 --dataset  TieredImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.03,0.1,0.3,1,3,10,30"
```

Output
<pre>
valid_SupervisedAcc                                0.8083+0.0068 (ep5)   0.8186+0.0065 (ep0)   Mean    0.8140 
test_SupervisedAcc                                 0.8343+0.0065 (ep5)   <b>0.8424+0.0065</b> (ep0)   Mean    0.8363
valid_UnsupervisedAcc_softmax_reg3                 0.7020+0.0112 (ep6)   0.7265+0.0101 (ep2)   Mean    0.7148
test_UnsupervisedAcc_softmax_reg3                  0.7542+0.0105 (ep6)   <b>0.7536+0.0104</b> (ep2)   Mean    0.7559
valid_ClusteringAcc_softmax_reg3                   0.7683+0.0098 (ep6)   0.7968+0.0095 (ep0)   Mean    0.7810 
test_ClusteringAcc_softmax_reg3                    0.8099+0.0100 (ep6)   0.8249+0.0097 (ep0)   Mean    0.8141
valid_ProtoTransductiveProbAcc_reg3                0.8278+0.0067 (ep5)   0.8409+0.0064 (ep0)   Mean    0.8339 
test_ProtoTransductiveProbAcc_reg3                 0.8556+0.0062 (ep5)   0.8635+0.0063 (ep0)   Mean    0.8567
valid_ProtoTransductiveDstAcc_reg3                 0.8278+0.0067 (ep5)   0.8409+0.0064 (ep0)   Mean    0.8339
test_ProtoTransductiveDstAcc_reg3                  0.8556+0.0062 (ep5)   0.8635+0.0063 (ep0)   Mean    0.8567
</pre>



## Cross-Domain Experiments

### miniImageNet to CUB - Conv-4

```bash
python train_fsl.py  --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet2CUB --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1 --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.1,0.3,1,3,10"
```

Output
<pre>
valid_SupervisedAcc                                0.6615+0.0072 (ep0)   0.6845+0.0077 (ep7)   Mean    0.6751 
test_SupervisedAcc                                 0.6108+0.0078 (ep0)   <b>0.6252+0.0073</b> (ep7)   Mean    0.6196
valid_UnsupervisedAcc_sinkhorn_reg3                0.5140+0.0093 (ep0)   0.5436+0.0103 (ep7)   Mean    0.5257
test_UnsupervisedAcc_sinkhorn_reg3                 0.4539+0.0093 (ep0)   <b>0.4701+0.0091</b> (ep7)   Mean    0.4666
valid_ClusteringAcc_softmax_reg3                   0.6314+0.0086 (ep0)   0.6535+0.0098 (ep7)   Mean    0.6388 
test_ClusteringAcc_softmax_reg3                    0.5887+0.0082 (ep0)   0.5972+0.0087 (ep7)   Mean    0.5937
valid_ProtoTransductiveProbAcc_reg3                0.6786+0.0074 (ep0)   0.7020+0.0077 (ep7)   Mean    0.6936 
test_ProtoTransductiveProbAcc_reg3                 0.6279+0.0079 (ep0)   0.6390+0.0076 (ep7)   Mean    0.6363
valid_ProtoTransductiveDstAcc_reg3                 0.6786+0.0074 (ep0)   0.7020+0.0077 (ep7)   Mean    0.6936
test_ProtoTransductiveDstAcc_reg3                  0.6279+0.0079 (ep0)   0.6390+0.0076 (ep7)   Mean    0.6363
</pre>

### miniImageNet to CUB - ResNet-12

```bash
python train_fsl.py  --max_epoch 200 --model_class ProtoNet  --backbone_class Res12 --dataset MiniImageNet2CUB --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean --tst_free 1 --tst_criterion UnsupervisedAcc_softmax --sinkhorn_reg "0.1,0.3,1,3,10"
```

Output
<pre>
valid_SupervisedAcc                                0.6605+0.0074 (ep1)   0.6698+0.0074 (ep4)   Mean    0.6635 
test_SupervisedAcc                                 0.6089+0.0077 (ep1)   <b>0.6138+0.0076</b> (ep4)   Mean    0.6113
valid_UnsupervisedAcc_softmax_reg10                0.5019+0.0096 (ep6)   0.5180+0.0094 (ep5)   Mean    0.5092 
test_UnsupervisedAcc_softmax_reg10                 0.4474+0.0091 (ep6)   <b>0.4462+0.0090</b> (ep5)   Mean    0.4485
valid_ClusteringAcc_softmax_reg10                  0.6276+0.0088 (ep6)   0.6377+0.0089 (ep5)   Mean    0.6322
test_ClusteringAcc_softmax_reg10                   0.5842+0.0081 (ep6)   0.5809+0.0085 (ep5)   Mean    0.5838
valid_ProtoTransductiveProbAcc_reg10               0.6774+0.0076 (ep3)   0.6883+0.0076 (ep4)   Mean    0.6815 
test_ProtoTransductiveProbAcc_reg10                0.6304+0.0078 (ep3)   0.6292+0.0077 (ep4)   Mean    0.6271
valid_ProtoTransductiveDstAcc_reg10                0.6774+0.0076 (ep3)   0.6883+0.0076 (ep4)   Mean    0.6815
test_ProtoTransductiveDstAcc_reg10                 0.6304+0.0078 (ep3)   0.6292+0.0077 (ep4)   Mean    0.6271
</pre>


<br><br><br><br>

# Original README

The code repository for "[Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions](https://arxiv.org/abs/1812.03664)" (Accepted by CVPR 2020) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @inproceedings{ye2020fewshot,
      author    = {Han-Jia Ye and
                   Hexiang Hu and
                   De-Chuan Zhan and
                   Fei Sha},
      title     = {Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year      = {2020}
    }

## Embedding Adaptation with Set-to-Set Functions

We propose a novel model-based approach to adapt the instance embeddings to the target classification task with a #set-to-set# function, yielding embeddings that are task-specific and are discriminative. We empirically investigated various instantiations of such set-to-set functions and observed the Transformer is most effective --- as it naturally satisfies key properties of our desired model. We denote our method as Few-shot Embedding Adaptation with Transformer (FEAT).

<img src='imgs/architecture.png' width='640' height='280'>

## Standard Few-shot Learning Results

Experimental results on few-shot learning datasets with ResNet-12 backbone (Same as [this repo](https://github.com/kjunelee/MetaOptNet)). We report average results with 10,000 randomly sampled few-shot learning episodes for stablized evaluation.

**MiniImageNet Dataset**
|  Setups  | 1-Shot 5-Way | 5-Shot 5-Way |   Link to Weights |
|:--------:|:------------:|:------------:|:-----------------:|
| ProtoNet |     62.39    |     80.53    | [Coming Soon]() |
|  BILSTM  |     63.90    |     80.63    | [Coming Soon]() |
| DEEPSETS |     64.14    |     80.93    | [Coming Soon]() |
|    GCN   |     64.50    |     81.65    | [Coming Soon]() |
|   FEAT   |   **66.78**  |   **82.05**  | [Coming Soon]() |

**TieredImageNet Dataset**

|  Setups  | 1-Shot 5-Way | 5-Shot 5-Way |   Link to Weights |
|:--------:|:------------:|:------------:|:-----------------:|
| ProtoNet |     68.23    |     84.03    | [Coming Soon]() |
|  BILSTM  |     68.14    |     84.23    | [Coming Soon]() |
| DEEPSETS |     68.59    |     84.36    | [Coming Soon]() |
|    GCN   |     68.20    |     84.64    | [Coming Soon]() |
|   FEAT   |   **70.80**  |   **84.79**  | [Coming Soon]() |

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images

- Pre-trained weights: please download the [pre-trained weights](https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE) of the encoder if needed. The pre-trained weights can be downloaded in a [zip file](https://drive.google.com/file/d/1XcUZMNTQ-79_2AkNG3E04zh6bDYnPAMY/view?usp=sharing).

## Dataset

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively.

### CUB Dataset
[Caltech-UCSD Birds (CUB) 200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) is initially designed for fine-grained classification. It contains in total 11,788 images of birds over 200 species. On CUB, we randomly sampled 100 species as SEEN classes, and another two 50 species are used as two UNSEEN sets. We crop all images with given bounding boxes before training. We only test CUB with the ConvNet backbone in our work.

### TieredImageNet Dataset
[TieredImageNet](https://github.com/renmengye/few-shot-ssl-public) is a large-scale dataset  with more categories, which contains 351, 97, and 160 categoriesfor model training, validation, and evaluation, respectively. The dataset can also be download from [here](https://github.com/kjunelee/MetaOptNet).
We only test TieredImageNet with ResNet backbone in our work.

Check [this](https://github.com/Sha-Lab/FEAT/blob/master/data/README.md) for details of data downloading and preprocessing.

## Code Structures
To reproduce our experiments with FEAT, please use **train_fsl.py**. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`: Images and splits for the data sets.
 - `saves`: The pre-trained weights of different networks.
 - `checkpoints`: To save the trained models.

## Model Training and Evaluation
Please use **train_fsl.py** and follow the instructions below. FEAT meta-learns the embedding adaptation process such that all the training instance embeddings in a task is adapted, based on their contextual task information, using Transformer. The file will automatically evaluate the model on the meta-test set with 10,000 tasks after given epochs.

## Arguments
The train_fsl.py takes the following command line options (details are in the `model/utils.py`):

**Task Related Arguments**
- `dataset`: Option for the dataset (`MiniImageNet`, `TieredImageNet`, or `CUB`), default to `MiniImageNet`

- `way`: The number of classes in a few-shot task during meta-training, default to `5`

- `eval_way`: The number of classes in a few-shot task during meta-test, default to `5`

- `shot`: Number of instances in each class in a few-shot task during meta-training, default to `1`

- `eval_shot`: Number of instances in each class in a few-shot task during meta-test, default to `1`

- `query`: Number of instances in each class to evaluate the performance during meta-training, default to `15`

- `eval_query`: Number of instances in each class to evaluate the performance during meta-test, default to `15`

**Optimization Related Arguments**
- `max_epoch`: The maximum number of training epochs, default to `200`

- `episodes_per_epoch`: The number of tasks sampled in each epoch, default to `100`

- `num_eval_episodes`: The number of tasks sampled from the meta-val set to evaluate the performance of the model (note that we fix sampling 10,000 tasks from the meta-test set during final evaluation), default to `200`

- `lr`: Learning rate for the model, default to `0.0001` with pre-trained weights

- `lr_mul`: This is specially designed for set-to-set functions like FEAT. The learning rate for the top layer will be multiplied by this value (usually with faster learning rate). Default to `10`

- `lr_scheduler`: The scheduler to set the learning rate (`step`, `multistep`, or `cosine`), default to `step`

- `step_size`: The step scheduler to decrease the learning rate. Set it to a single value if choose the `step` scheduler and provide multiple values when choosing the `multistep` scheduler. Default to `20`

- `gamma`: Learning rate ratio for `step` or `multistep` scheduler, default to `0.2`

- `fix_BN`: Set the encoder to the evaluation mode during the meta-training. This parameter is useful when meta-learning with the WRN. Default to `False`

- `augment`: Whether to do data augmentation or not during meta-training, default to `False`

- `mom`: The momentum value for the SGD optimizer, default to `0.9`

- `weight_decay`: The weight_decay value for SGD optimizer, default to `0.0005`

**Model Related Arguments**
- `model_class`: The model to use during meta-learning. We provide implementations for baselines (`MatchNet` and `ProtoNet`), set-to-set functions (`BILSTM`, `DeepSet`, `GCN`, and our `FEAT`). We also include an instance-specific embedding adaptation approach `FEAT`, which is discussed in the old version of the paper. Default to `FEAT`

- `use_euclidean`: Use the euclidean distance or the cosine similarity to compute pairwise distances. We use the euclidean distance in the paper. Default to `False`

- `backbone_class`: Types of the encoder, i.e., the convolution network (`ConvNet`), ResNet-12 (`Res12`), or Wide ResNet (`WRN`), default to `ConvNet`

- `balance`: This is the balance weight for the contrastive regularizer. Default to `0`

- `temperature`: Temperature over the logits, we #divide# logits with this value. It is useful when meta-learning with pre-trained weights. Default to `1`

- `temperature2`: Temperature over the logits in the regularizer, we divide logits with this value. This is specially designed for the contrastive regularizer. Default to `1`

**Other Arguments** 
- `orig_imsize`: Whether to resize the images before loading the data into the memory. `-1` means we do not resize the images and do not read all images into the memory. Default to `-1`

- `multi_gpu`: Whether to use multiple gpus during meta-training, default to `False`

- `gpu`: The index of GPU to use. Please provide multiple indexes if choose `multi_gpu`. Default to `0`

- `log_interval`: How often to log the meta-training information, default to every `50` tasks

- `eval_interval`: How often to validate the model over the meta-val set, default to every `1` epoch

- `save_dir`: The path to save the learned models, default to `./checkpoints`

Running the command without arguments will train the models with the default hyper-parameter values. Loss changes will be recorded as a tensorboard file.

## Training scripts for FEAT

For example, to train the 1-shot/5-shot 5-way FEAT model with ConvNet backbone on MiniImageNet:

    $ python train_fsl.py  --max_epoch 200 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 8 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1
    $ python train_fsl.py  --max_epoch 200 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 14 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1

to train the 1-shot/5-shot 5-way FEAT model with ResNet-12 backbone on MiniImageNet:

    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean
    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean

to train the 1-shot/5-shot 5-way FEAT model with ResNet-12 backbone on TieredImageNet:

    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1  --use_euclidean
    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1  --use_euclidean

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)

- [MatchingNet](https://github.com/gitabcworld/MatchingNetworks)

- [PFA](https://github.com/joe-siyuan-qiao/FewShot-CVPR/)

- [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

- [MetaOptNet](https://github.com/kjunelee/MetaOptNet/)

