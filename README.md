# SCNet

This repository is the official implementation of [CrowdioSet and PaRIRset: Two Datasets Towards Live Music Source  Separation](http://arxiv.org/abs/2607.27828)

![pipeline](images/pipeline.png)

---

## Preparing the data (MOISESDB + MUSDB18HQ + CROWDIOSET + PARIRSET)

Download [MOISESDB dataset](https://music.ai/research/)
Install HF CLI `curl -LsSf https://hf.co/cli/install.sh | bash`
Download the remaining datasets and unzip:

```bash
curl -L -C - -o musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
unzip musdb18hq.zip -d musdb18hq
unzip moisesdb.zip -d moisesdb
hf download enricguso/crowdioset --repo-type dataset --local-dir crowdioset
hf download enricguso/parirset --repo-type dataset --local-dir parirset
```

Then, you need to install the MOISESDB parser requirements.

```bash
pip install git+https://github.com/moises-ai/moises-db.git
```

Then run `prepare_musdbmoises.py`, which merges MUSDB18HQ, MOISESDB and the CrowdioSet singalongs into an output `musdbmoises` dir:

```bash
python prepare_musdbmoises.py --moises_path moisesdb --musdb_path musdb18hq --out_path musdbmoises --crowdioset_path crowdioset
```


## Install additional requirements for SCNet inference or training
```bash
pip install -r requirements.txt 
```

## Training

Default `noisy` model (MUSDBMOISES+CrowdioSet w/o reverb, the one from the subjective eval) is trained:
```bash
python3 scnet/train_noisy.py --config_path conf/ismir26_noisy.yaml --save_path path/to/save/checkpoint/
```
To train a `noisyrev` model on (MUSDBMOISES+CrowdioSet)*PaRIRset use:
```bash
python3 scnet/train_noisy.py --config_path conf/ismir26_noisyrev.yaml --save_path path/to/save/checkpoint/
```
To train the `clean` model (MUSDBMOISES+CrowdioSet):
```bash
python3 scnet/train_clean.py --config_path conf/ismir26_clean.yaml --save_path path/to/save/checkpoint/
```
To train the `rev` reverberant model (MUSDBMOISES+CrowdioSet)*PaRIRset:
```bash
python3 scnet/train_clean.py --config_path conf/ismir26_rev.yaml --save_path path/to/save/checkpoint/
```
---

## Inference

The model checkpoints are found under `/models` and can be used with the provided `scnet/inference.py` script, which selects a checkpoint with the arguments `--noisy` and `--reverberant`.

Inference with the default `noisy` model (MUSDBMOISES+CrowdioSet w/o reverb, the one from the subjective eval) would be:
```
python scnet/inference.py --noisy --mix_path /path/to/mixture/audio/file --out_path /path/where/to/store/estimates 
```

To infer with the `clean` model trained on (MUSDBMOISES) use:
```
python scnet/inference.py --mix_path /path/to/mixture/audio/file --out_path /path/where/to/store/estimates 
```

To infer with the `rev` model (MUSDBMOISES * PaRIRset):
```
python scnet/inference.py --reverberant --mix_path /path/to/mixture/audio/file --out_path /path/where/to/store/estimates 
```

To infer with the `noisyrev` model (MUSDBMOISES+CrowdioSet)*PaRIRset:
```
python scnet/inference.py --noisy --reverberant --mix_path /path/to/mixture/audio/file --out_path /path/where/to/store/estimates 
```

## Inference

To reproduce the paper results and objective evaluation, take `scnet/crowdioset_eval.py` (might need some tweaking).

---

## Citing



If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{guso2026crowdioset,
  title={CrowdioSet and PaRIRset: Two Datasets Towards Live Music Source Separation},
  author={Gus{\'o}, Enric and Serra, Xavier},
  booktitle={Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2026}
}
```

This work is based on:
```bibtex
@misc{tong2024scnet,
      title={SCNet: Sparse Compression Network for Music Source Separation}, 
      author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
      year={2024},
      eprint={2401.13276},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```


