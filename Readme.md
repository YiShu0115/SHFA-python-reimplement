# Python SHFA
This is a personal Python reimplement of HFA and SHFA[1]. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the  <a href="https://github.com/wenli-vision/SHFA_release#license">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately [2].

For the original Matlab version of this work, please see: https://github.com/wenli-vision/SHFA_release.

## Requirements
Our code is developed mainly on python 3.8, scikit-learn 1.1.3, scikit-image 0.19.3.

```shell
conda create --name SHFA python=3.8
conda activate SHFA
pip install -r requirements.txt
```

## Usage
We provide inference examples `infer_HFA.py` and `infer_SHFA.py` seperately for HFA and SHFA. By default, Amazon dataset is used as source domain with 800 dimensions and DSLR dataset is used as target domain with 600 dimensions. 

You can test the method by:
```Shell
python infer_HFA.py
```
```Shell
python infer_SHFA.py
```


If the process goes smoothly, the output accuracy of HFA and SHFA would be around 0.582716 and 0.570370 seperately.

## references

```
[1] @article{li_learning_2014,
	title = {Learning {With} {Augmented} {Features} for {Supervised} and {Semi}-{Supervised} {Heterogeneous} {Domain} {Adaptation}},
	volume = {36},
	copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
	issn = {0162-8828, 2160-9292},
	url = {http://ieeexplore.ieee.org/document/6587717/},
	doi = {10.1109/TPAMI.2013.167},
	language = {en},
	number = {6},
	urldate = {2025-04-08},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	author = {Li, Wen and Duan, Lixin and Xu, Dong and Tsang, Ivor W.},
	month = {06},
	year = {2014},
	pages = {1134--1148}

}
```

## Acknowledgements
This project relies on code from existing repositories: [SHFA](https://github.com/wenli-vision/SHFA_release). We thank the original authors for their excellent work. If you have any problem please contact yezi.cai0115@gmail.com.
