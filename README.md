# Physical Design Watermarking 

Artifact evaluation for MLCAD 2024 paper "Automated Physical Design Watermarking Leveraging Graph Neural Networks" [Paper](https://arxiv.org/abs/2407.20544)

Code for TCAD 2025 paper "ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection" [Paper](https://arxiv.org/abs/2404.18407)

#### Environment Setup

```bash 
# clone code
git clone --recursive https://github.com/ruisizhang123/PD_WM_GNN.git
cd PD_WM_GNN
# create conda env
conda create -n  pd_wm_gnn python=3.9
conda activate pd_wm_gnn
# install required packages
pip install -r requirement.txt
# build
bash build.sh
```

Our codebase builds heavily upon [DREAMPlace](https://github.com/limbo018/DREAMPlace). If the environment missed some packages, you can alternatively (1) look for instructions from DREAMPlace repo or (2) build with dock file as follows:

```bash
docker build . --file Dockerfile --tag PD_WM_GNN/dreamplace:cuda
```

1. Download required benchmarks

```bash
cd benchmarks
python ispd2005_2015.py
python ispd2019.py
```

#### Watermark layout with heuristic approach (TCAD 2025)
2. Watermark design

Watermark ISPD19 test1 design, with our heuristic approach.

```bash
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/combine_wm.json  # For ICMarks
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/global_wm.json   # For Global Watermarking
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/detail_wm.json   # For Detail Watermarking
```

We use [CU-GR](https://github.com/cuhk-eda/cu-gr) to route the wm'ed layout. We also provide the pre-built binary software to evaluate the wirelength in `iccad19gr_upd`.

```bash
./iccad19gr_upd -lef  benchmarks/ispd2019/ispd19_test1/ispd19_test1.input.lef -def  results/ispd19_test1.input/ispd19_test1.input.500.def -output result.solution.guide -threads 8  >> results/ispd19_test1.input/ispd19_test1/log.txt
```

#### Watermark layout with GNN (MLCAD 2024)


2. Watermark design (Inference)

Watermark ISPD19 test1 design, with our pre-trained GNN model.

```bash
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/graph.json 
```

We use [CU-GR](https://github.com/cuhk-eda/cu-gr) to route the wm'ed layout. 

```bash
./iccad19gr_upd -lef  benchmarks/ispd2019/ispd19_test1/ispd19_test1.input.lef -def  results/ispd19_test1.input/ispd19_test1.input.500.def -output result.solution.guide -threads 8  >> results/ispd19_test1.input/ispd19_test1/log.txt
```

3. Train GNN from scratch

Change `mode` in `./test/graph.json` from `inference` to `train`. We collect and train our model on ISPD19 test6 design.

```bash
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test6.json ./test/graph.json 
```

4. Attack Evaluation

Change `attack` in `./test/graph.json` from `0` to `1`. 

```bash
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/graph.json 
```

#### Citation

If you found our code/paper helpful, please kindly cite:

```latex 
@article{zhang2025icmarks,
  title={Icmarks: A robust watermarking framework for integrated circuit physical design ip protection},
  author={Zhang, Ruisi and Rajarathnam, Rachel Selina and Pan, David Z and Koushanfar, Farinaz},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2025},
  publisher={IEEE}
}

@inproceedings{zhang2024automated,
  title={Automated Physical Design Watermarking Leveraging Graph Neural Networks},
  author={Zhang, Ruisi and Rajarathnam, Rachel Selina and Pan, David Z and Koushanfar, Farinaz},
  booktitle={Proceedings of the 2024 ACM/IEEE International Symposium on Machine Learning for CAD},
  pages={1--10},
  year={2024}
}
```
