# EIT: Embedded Interaction Transformer
PyTorch implementation of "[Seeing the forest and the tree: Building 
representations of both individual and collective dynamics with transformers](https://arxiv.org/pdf/2206.06131.pdf)" (**NeurIPS 2022**).


## Instructions

##### Synthetic experiments

Use [synthetic_data_threebody.py](https://github.com/nerdslab/EIT/blob/main/synthetic_data_threebody.py) or 
[synthetic_data_twobody.py](https://github.com/nerdslab/EIT/blob/main/synthetic_data_twobody.py)
to generate synthetic data, and then use the corresponding synthetic_exp_EXP.py to run experiments.
The models are defined in [synthetic_exp_twobody.py](https://github.com/nerdslab/EIT/blob/main/synthetic_exp_twobody.py).

##### Neural activity experiments




## Code contributors

- Ran Liu (Maintainer), github: ranliu98

- Jingyun Xiao, github: jingyunx

- Mehdi Azabou , github: mazabou



## Citation
If you find the code useful for your research, please consider citing our work:

```
@inproceedings{liu2022seeing,
  title={Seeing the forest and the tree: Building representations of both individual and collective dynamics with transformers},
  author={Liu, Ran and Azabou, Mehdi and Dabagia, Max and Xiao, Jingyun and Dyer, Eva L},
  journal={arXiv preprint arXiv:2206.06131},
  year={2022}
}
```

AND [SwapVAE](https://github.com/nerdslab/SwapVAE)

```
@article{liu2021drop,
  title={Drop, Swap, and Generate: A Self-Supervised Approach for Generating Neural Activity},
  author={Liu, Ran and Azabou, Mehdi and Dabagia, Max and Lin, Chi-Heng and Gheshlaghi Azar, Mohammad and Hengen, Keith and Valko, Michal and Dyer, Eva},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={10587--10599},
  year={2021}
}
```
