<h1 align="center"><b>Direct parsing to sentiment graphs (WIP)</b></h1>

<p align="center">
  <i><b>David Samuel, Jeremy Barnes, Robin Kurtz, Stephan Oepen, Lilja Øvrelid and Erik Velldal</b></i>
</p>

<p align="center">
  <i>
    University of Oslo, Language Technology Group<br>
    University of the Basque Country UPV/EHU, HiTZ Center – Ixa<br>
    National Library of Sweden, KBLab
  </i>
</p>
<br>

<p align="center">
  <a href="TODO"><b>Paper</b></a><br>
  <a href="TODO"><b>Pretrained models</b></a><br>
  <a href="TODO"><b>Interactive demo on Google Colab</b></a>
</p>

<p align="center">
  <img src="img/architecture_inference.png" alt="Overall architecture" width="340"/>  
</p>

_______

<br>

This repository provides the official PyTorch implementation of our paper "Direct parsing to sentiment graphs" together with [pretrained *base* models](https://drive.google.com/drive/folders/11ozu_uo9z3wJwKl1Ei2C3aBNUvb66E-2?usp=sharing) for all six datasets (TODO): Darmstadt, MPQA, Multibooked_ca, Multibooked_eu and NoReC.

_______

<br>

## How to run

### :feet: &nbsp; Training

To train PERIN on NoReC, run the following script. Other configurations are located in the `perin/config` folder.
```sh
cd perin
sbatch run.sh config/seq_norec.yaml
```

### :feet: &nbsp; Inference

You can run the inference on the validation and test datasets by running:
```sh
python3 inference.py --checkpoint "path_to_pretrained_model.h5" --data_directory ${data_dir}
```

## Citation

```
TBA
```
