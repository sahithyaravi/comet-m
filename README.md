# COMET-M: Reasoning about multiple events in complex sentences
### Install Requirements
- Create a virtual env.
```
virtualenv -p python3 --no-download comet
source comet/bin/activate 
```
- Install requirements.
```
pip3 install -r requirements.txt
pip3 install gitpython
pip3 install torch
pip3 install transformers
```

### MEI Dataset
The datasets used for training COMET-M versions are available [here](https://drive.google.com/drive/folders/1g4dlgXQANI3kPEAaMAA3ttS-NaGrBSr8?usp=share_link).
- MEI - The human-annotated MEI dataset containing gold multi-event inferences.
- M-NLI - Silver standard Inferences generated from COMET, and filtered using NLI to remove contradictions.
- M-Mimic - Silver standard Inferences generated generated from GPT 3.5 Turbo.

### Checkpoints
If you directly want to use the checkpoints of the trained models, you can download them on huggingface [COMET-M-MEI](https://huggingface.co/sahithyaravi/comet-m), [COMET-M-NLI](https://huggingface.co/sahithyaravi/comet-m-nli) and [COMET-M-MIMIC](https://huggingface.co/sahithyaravi/comet-m-mimic). An example of how to generate inferences from these models is provided in `multi-event/inference_tests.py`.

### Finetuning
- If you want to start from the COMET checkpoint, get the pretrained comet from [here](https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip) and unzip/link it in the root folder. To start off with BART, you can simply specify model_path as `facebook/bart-large`.

- To train COMET on any of the above datasets, run `scripts/train.sh` and set the 'train_path' appropriately.

### Citation
```
@inproceedings{ravi-etal-2023-comet,
    title = "{COMET}-{M}: Reasoning about Multiple Events in Complex Sentences",
    author = "Ravi, Sahithya  and
      Ng, Raymond  and
      Shwartz, Vered",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.861",
    pages = "12921--12937",
}
```



