# Citation

## If you use this project

```bibtex
@misc{trencodeswitchlid2026,
  title        = {TR-EN-CodeSwitch-LID: Generated Training Data for
                  Turkish-English Code-Mixed Language Identification},
  author       = {{The Avengers (TR-EN-CodeSwitch-LID team)}},
  year         = {2026},
  howpublished = {\url{https://github.com/ShahemHassoun/TR-EN-CodeSwitch-LID}},
  note         = {NLP course project}
}
```

## Please also cite what we built on

### TurEngMix (benchmark data, splits, scoring code and LLM prompts)

Ilayda Dogan, Phuong-Anh Nguyen-Le and Julia Mendelsohn. 2026.
*TurEngMix: A Text Corpus and Benchmark for Turkish-English Code-Mixed Language
Identification and Named Entity Recognition.* University of Maryland. arXiv:2609.06963.
Code and data: https://github.com/kemnguyenle/TurEngMix

```bibtex
@misc{dogan2026turengmix,
  title         = {TurEngMix: A Text Corpus and Benchmark for Turkish-English
                   Code-Mixed Language Identification and Named Entity Recognition},
  author        = {Dogan, Ilayda and Nguyen-Le, Phuong-Anh and Mendelsohn, Julia},
  year          = {2026},
  eprint        = {2609.06963},
  archivePrefix = {arXiv},
  url           = {https://github.com/kemnguyenle/TurEngMix}
}
```

### BERTurk (model `dbmdz/bert-base-turkish-cased`)

Stefan Schweter. 2020. *BERTurk: BERT models for Turkish.* Zenodo.
https://huggingface.co/dbmdz/bert-base-turkish-cased

```bibtex
@software{schweter2020berturk,
  author    = {Schweter, Stefan},
  title     = {BERTurk - BERT models for Turkish},
  year      = {2020},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.3770924},
  url       = {https://doi.org/10.5281/zenodo.3770924}
}
```

### XLM-R (model `FacebookAI/xlm-roberta-base`)

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek,
Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. 2020.
*Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
https://huggingface.co/FacebookAI/xlm-roberta-base

```bibtex
@inproceedings{conneau2020unsupervised,
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  author    = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and
               Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and
               Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for
               Computational Linguistics},
  year      = {2020},
  pages     = {8440--8451}
}
```

### Zemberek (Turkish morphology, used in the rule baseline)

Ahmet Afşın Akın and Mehmet Dündar Akın. 2007. *Zemberek, an open source NLP framework for
Turkic languages.* https://github.com/ahmetaa/zemberek-nlp
(Python port used here: zemberek-python, https://github.com/Loodos/zemberek-python)

### Other tools

- **wordfreq** (English/Turkish word frequencies, rule baseline and data checks): Robyn Speer.
  https://github.com/rspeer/wordfreq
- **sklearn-crfsuite** (CRF baseline): https://github.com/TeamHG-Memex/sklearn-crfsuite
- **Hugging Face Transformers** (fine-tuning): Wolf et al. 2020, EMNLP System Demonstrations.
  https://github.com/huggingface/transformers
- **GPT-4o** (prompted baseline): OpenAI, accessed through the OpenAI API, 2026.
- **Claude** (Anthropic): used to write the generated sentences, see `data/generated/README.md`.
