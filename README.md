# Densely Interactive Inference Network (DIIN)

This is the code to reproduce the model in [Natural Language Inference over Interaction Space](https://arxiv.org/abs/1709.04348).

## Environment
	python 3.6
	tensorflow = 1.3


## Setup
	$ git clone https://github.com/YichenGong/Densely-Interactive-Inference-Network.git
	$ cd Densely-Interactive-Inference-Network
	$ pip install -r requirements.txt

## Download Data
First, run `download.py` for the datasets and the preprocessed file:

```
$ cd data
$ python download.py
```

Then, manually download download MultiNLI 0.9
[matched](https://www.kaggle.com/c/multinli-matched-open-evaluation/data)
and [mismatched](https://www.kaggle.com/c/multinli-mismatched-open-evaluation/data)
test set under data/multinli_0.9 folder

If any of the auto download fails, you can manually download them from:
* [SNLI 1.0](https://nlp.stanford.edu/projects/snli/)
* [MultiNLI 0.9](https://www.nyu.edu/projects/bowman/multinli/)
* [Glove](https://nlp.stanford.edu/projects/glove/)
* [shared.json](https://drive.google.com/file/d/0B6CTyAhSHoJTa3ZSSE5QQUJrb3M/view?usp=sharing)


When you finish downloading, your data folder should look like this:

```
    $ tree data
    data
    ├── download.py
    ├── glove.840B.300d.txt
    ├── multinli_0.9
    │   ├── Icon\015
    │   ├── multinli_0.9_dev_matched.jsonl
    │   ├── multinli_0.9_dev_matched.txt
    │   ├── multinli_0.9_dev_mismatched.jsonl
    │   ├── multinli_0.9_dev_mismatched.txt
    │   ├── multinli_0.9_test_matched_sample_submission.csv
    │   ├── multinli_0.9_test_mismatched_sample_submission.csv
    │   ├── multinli_0.9_train.jsonl
    │   ├── multinli_0.9_train.txt
    │   └── paper.pdf
    ├── shared.json
    └── snli_1.0
        ├── Icon\015
        ├── README.txt
        ├── snli_1.0_dev.jsonl
        ├── snli_1.0_dev.txt
        ├── snli_1.0_test.jsonl
        ├── snli_1.0_test.txt
        ├── snli_1.0_train.jsonl
        └── snli_1.0_train.txt
```

I don't recommend you to use multinli_1.0 here because the id doesn't match the id in preprocessed sample id.

## To run the code
	$ cd python 
	# on MultiNLI
	$ python train_mnli.py DIIN demo_testing 
	# on SNLI
	$ python train_mnli.py DIIN demo_testing_SNLI --training_completely_on_snli




## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

