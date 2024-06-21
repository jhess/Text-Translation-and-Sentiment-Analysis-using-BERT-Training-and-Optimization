# Text Translation and Sentiment Analysis using Transformer Training and Optimization

This project uses a MarianMTModel Transformer "Helsinki-NLP" model from HuggingFace's model library to translate French and Spanish text to English. It uses the "distilbert-base-uncased" transformer that is subsequently optimized to perform sentiment analysis on the translated text.

## Table of Contents
- [Installation](#installation)
- [Setup](#setup)
- [Running and Troubleshooting](#running-and-troubleshooting)
- [Release Compatibility Matrix](#release-compatibility-matrix)

## Installation

Install the dependencies from the requirements.txt using the command below:
```shell
pip install -r requirements.txt
```

This project uses PyTorch 2.3.0 CUDA 11.8 support (previoulsy used 2.3.1+CPU)

If there are issues installing the torch versions with CUDA support enabled, you will need to run `pip install` on each of them separately in the command line: torch and torchvision:
```shell
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
```shell
pip install torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Setup

If you get the below error: <br />
`ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'venv\\Lib\\site-packages\\~orch\\lib\\asmjit.dll' Check the permissions.` <br />
You have to mark the ~orch folder under site-packages as writeable (not read-only).

If you get this error: <br />
`WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1006)'))': /packages/0a/16/c83877524c47976f16703d2e05c363244bc1e60ab439e078b3cd046d07db/pillow-10.3.0-cp311-cp311-win_amd64.whl.metadata`

You have to add `--trusted-host files.pythonhosted.org` when installing torchvision with CUDA support:
```shell
pip install --trusted-host files.pythonhosted.org torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```


Follow [these](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models) steps to download root CA certificate from Huggingface's website.

Then add this code as cell to use the certificate you downloaded as an enrivonment variable (this is dymanic, has to be done every time):

```python
import os

os.environ['REQUESTS_CA_BUNDLE'] = '<PATH-TO-CA.crt-FILE>'
```

NOTE: You may need to add two CA cert files to the CA environment variable: one for HuggingFace, and one for [datasets](https://raw.githubusercontent.com/huggingface/datasets) (*NOTE: requests library seems to have a bug where it doesn't recognize environment variables with more than one path separated by the standard convention of the ";" delimeter and no spaces, and you may get an OSError: `OSError: Could not find a suitable TLS CA certificate bundle, invalid path:`):
```python
import os

path1 = '<PATH-TO-CA.crt-FILE1>'
path2 = '<PATH-TO-CA.crt-FILE2>'

bundlePath = os.pathsep.join([path1, path2]) # same as ";".join

os.environ['REQUESTS_CA_BUNDLE'] = bundlePath
```

When attempting to run the transformer notebook, you may get an error that sentencepiece needs to be also installed along with torch. If you get an SSL error downloading it via pip from pythonhosted.org, run this pip command (may need to run VS code in admin mode):
```shell
pip install --trusted-host=files.pythonhosted.org sentencepiece
```

Use this [link](https://pytorch.org/get-started/previous-versions/) to see what versions of CUDA enabled PyTorch to install.

## Running and Troubleshooting

You may get this error below when running trainer.Train(): <br />
`ImportError: Using the Trainer with PyTorch requires accelerate>=0.21.0: Please run pip install transformers[torch] or pip install accelerate -U`
It is not clear what fixes this. Installing accelerate separately and closing/reopening VS code may solve the issue.

For the HuggingFace transformer model to return a loss, it should be passed labels (i.e. the ground truth targets), besides the inputs (input_ids and attention_mask). If not, a ValueError will be returned when calling `trainer.train()` that says:
`ValueError: The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.`

It is recommended therefore to use Pandas dataframes for the train and test data to manually add the 'label' and 'text' column names, which will get converted to transformers library's datasets' Dataset object. Then, the text column values will be tokenized using a custom tokenize function. The 'label' column' values should only be binary integers, i.e. 0 or 1, or there will be more errors returned when calling `trainer.train()`.

Refer to the masked language modeling (MLM) script for details [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py).

The parameter `trust_source_code` may need to be set to true, as its behavior is not deterministic when running the trainer; sometimes it will work without this being set to true, other times it will be need to explicitly be set to true.
```python
def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy", trust_remote_code=True)
   load_f1 = load_metric("f1", trust_remote_code=True)
```

## Release Compatibility Matrix

This project uses Pytorch 2.3.0 CUDA 11.8 support (previoulsy used 2.3.1+CPU)

The following is the Release Compatibility Matrix for PyTorch releases:

| PyTorch version | Python | Stable CUDA | Experimental CUDA | Stable ROCm |
| --- | --- | --- | --- | --- |
| 2.3 | >=3.8, <=3.11, (3.12 experimental) | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 6.0 |
| 2.2 | >=3.8, <=3.11, (3.12 experimental) | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 5.7 |
| 2.1 | >=3.8, <=3.11 | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 | ROCm 5.6 |
| 2.0 | >=3.8, <=3.11 | CUDA 11.7, CUDNN 8.5.0.96 | CUDA 11.8, CUDNN 8.7.0.84 | ROCm 5.4 |
| 1.13 | >=3.7, <=3.10 | CUDA 11.6, CUDNN 8.3.2.44 | CUDA 11.7, CUDNN 8.5.0.96 | ROCm 5.2 |
| 1.12 | >=3.7, <=3.10 | CUDA 11.3, CUDNN 8.3.2.44 | CUDA 11.6, CUDNN 8.3.2.44 | ROCm 5.0 |
