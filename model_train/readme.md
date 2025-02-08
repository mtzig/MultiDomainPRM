## Environment setup

To setup the environment, run
```
pip install -r requirements.txt
```


## Training Data

Please download data from [this google drive link](https://drive.google.com/drive/folders/18HQlnkEfei7uh30eKUYF4EEj5UP0-7T6?usp=drive_link). There are two json files (one for PRM800K data and the other for MMLU-Pro-Train). Put them in the `\data` directory. 

## Running training

To do training of VersaPRM using our default configuration (requires 4 gpu's), run the following command:

```
./run_training.sh
```
We recommend using Deepspeed for data parallel training of model (which you can setup with `accelerate config`).