# Bechmark evaluation models of CMDC: Ef-Bi-LSTM and MulT

This is the bechmark evaluation code of the Chinese Multimodal Depression Corpus (CMDC).

The CMDC contains semi-structural interviews designed to support the screening and assessment of major depressive disorder in China. These interviews were collected as part of a larger effort to create automatic AI tools that interview people and identify visual, acoustic, and textual indicators of MDD. 

The Bi-LSTM code is partly borrowed from  https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextRNN.py and the MulT code is borrowed largely from https://github.com/yaohungt/Multimodal-Transformer. Thank them!

## Corpus Download Link

https://drive.google.com/drive/folders/1BnpE-xfQmRbQPWW4wxc_llMPkI2GZ2cy

The data are passcode protected. Please download and send the signed EULA to zoubochao@ustb.edu.cn for access request.

## Code Usage

### Prerequisties

<ul>
    <li>Pytorch</li>
    <li>Numpy</li>
    <li>Scipy</li>
    <li>Sklearn</li>
    <li>Pickle</li>
</ul>

### Preprocess

Before running the code, you need  to label the features of the three modalities of speech, text, and video. The specific data format is shown as below:

    
    {
    "train": {
        "text": [],          # text feature
        "audio": [],         # audio feature
        "vision": [],        # video feature
        "labels": []         # the phq score of each participant
        },
    "valid": {***},          # same as "train"
    "test": {***},           # same as "train"
    }
Then the tags file should be named as mosei_senti_data.pkl.

### How to run

There are two models (Bi-LSTM and MulT) defined in the models.py file.

The interface for the entire program is main.py. It works likes the following instruction:

    python main.py
