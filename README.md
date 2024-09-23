# PD_WM_GNN

Artifact evaluation for MLCAD 2024 paper "Automated Physical Design Watermarking Leveraging Graph Neural Networks"


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

#### Watermark layout

1. Download required benchmarks

```bash
cd benchmarks
python ispd2005_2015.py
python ispd2019.py
```

2. Watermark design (Inference)

Watermark ISPD19 test1 design, with our pre-trained GNN model.

```bash
python dreamplace/Placer.py test/ispd2019/lefdef/ispd19_test1.json ./test/graph.json 
```

We use [CU-GR](https://github.com/cuhk-eda/cu-gr) to route the wm'ed layout. Alternatively, you can use pre-built binary software to evaluate: 

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
@inproceedings{zhang2024automated,
  title={Automated Physical Design Watermarking Leveraging Graph Neural Networks},
  author={Zhang, Ruisi and Rajarathnam, Rachel Selina and Pan, David Z and Koushanfar, Farinaz},
  booktitle={Proceedings of the 2024 ACM/IEEE International Symposium on Machine Learning for CAD},
  pages={1--10},
  year={2024}
}
```