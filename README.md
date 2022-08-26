Here, we provide code for our two papers

"Convolutional Analysis Operator Learning by End-To-End Training of Iterative Neural Networks"

and 

"Convolutional Dictionary Learning by End-To-End Training of Iterative Neural Networks"

by A. Kofler, C. Wald,  T. Schaeffter, M. Haltmeier, C. Kolbitsch.

The papers are published in the proceedings of the IEEE ISBI 2022 conference and the IEEE EUSIPCO 2022 conference, respectively. 

Also, the accepted version of the paper can be found below:
https://arxiv.org/abs/2203.02166
https://arxiv.org/abs/2206.04447

The repository contains an implementation of both iterative network which uses convolutional sparsifying transforms. 
The networks can be used to learn the filters by end-to-end training.

## Generating the data-acquisition model

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/koflera/ConvSparsityNNs/main?labpath=basic_example.ipynb)

We provide an example of toy-data which can be used to get construct the data-acquisition model 
- img_320.npy:    -  the ground-truth image x
- ktraj_320.npy:  -  a set of k-space trajectories chosen according to the golden-angle method
- dcomp_320.npy:  -  the values of the employed density-compensation function
- csmap_320.npy:  -  coil-sensitvity maps for 12 receiver coils to be used in the operators

The image in the file img_320.npy was borrowed from http://www.cse.yorku.ca/~mridataset/.

Dependencies:
- PyTorch version 1.7.1 
- TorchKbNufft version 0.3.4 (https://github.com/mmuckley/torchkbnufft)

>> Further, note that in this implementation, the forward and adjoint NUFFT-operators are defined beforehand. 
Using this version of TorchKbNufft, this is required for using the classes MriSenseNufft/AdjMriSenseNufft. 
This means that, when training, only one set of k-space trajectories, density-compensation function and coil-sensitivity maps is used. 
This also means that the mini-batch size used for fine-tuning and testing has to be mb=1. 
However, at test time, you can of course set the csm, dcomp and ktraj according to the considered patient-specific setting.
This can in principle be circumvented by upgrading to the newest TorchKbNufft version, where csm can be used when calling the operators. 
In future, we might upgrade the code to be compatible with pytorch > 1.7.1 and TorchKbNufft > 0.3.4.

## Application of a previously trained NN-CAOL network:

Load the NN-CAOL model and use the network to reconstruct images:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/koflera/ConvSparsityNNs/main?labpath=nn_caol_test.ipynb)

## Comparison between the filters obtained by de-coupled (CAOL) pre-training vs. end-to-end trained filters (NN-CAOL)

Compare the performance of the reconstruction network when the used sparsifying filters were obtained by de-coupled pre-training (i.e. without including the data-acquisition model in the learning process) to the one where the sparsifying filters were obtained by supervised end-to-end training.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/koflera/ConvSparsityNNs/main?labpath=nn_caol_test.ipynb)

## Citing this work
If you find the code useful or you use it in your work, please cite our works:

@inproceedings{kofler2022nn_caol,
  title={Convolutional Analysis Operator Learning by End-to-End Training of Iterative Neural Networks},
  author={Kofler, Andreas and Wald, Christian and Schaeffter, Tobias and Haltmeier, Markus and Kolbitsch, Christoph},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}

@article{kofler2022nn_cdl,
  title={Convolutional Dictionary Learning by End-To-End Training of Iterative Neural Networks},
  author={Kofler, Andreas and Wald, Christian and Schaeffter, Tobias and Haltmeier, Markus and Kolbitsch, Christoph},
  journal={arXiv preprint arXiv:2206.04447},
  year={2022}
}
