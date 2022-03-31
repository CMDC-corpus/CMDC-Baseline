# Bechmark evaluation models of CMDC: Ef-LSTM and Mult

## Usage

### Prerequisties

<ul>
    <li>Pytorch</li>
    <li>Numpy</li>
    <li>Scipy</li>
    <li>Sklearn</li>
    <li>Pickle</li>
</ul>

### Preprocess

Before running the program, you need  to label the features of the three modalities of speech, text and video. The specific data format is shown in below.

    
    {
    "train": {
        "text": [],          # text feature
        "audio": [],         # audio feature
        "vision": [],        # video feature
        "labels": []         # the phq value of each participant
        },
    "valid": {***},          # same as "train"
    "test": {***},           # same as "train"
    }
Then the tags file should be named as mosei_senti_data.pkl.

### How to run

There are two models(lstm and mult) defined in the models.py file.And the interface file for the entire program is main.py.It works likes following instruction.

    python main.py
