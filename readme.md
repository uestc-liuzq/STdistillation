# Less is More: Efficient Time Series Dataset Condensation via Two-fold Modal Matching


## Requirements
python >= 3.8

For an express instillation, we include .yaml files.
   ```shell
   pip install requirements.yaml
   ```
## Datasets
Download data.
You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

## Generating Expert Trajectories
All the scripts are in the directory ```./scripts_buffer/```
The following command will train different dataset for 50 epochs each:
```
sh ./scripts_buffer/weather.sh
```
We used 50 epochs with the default learning rate for all of our experts.

## Time Series Dataset Condensation with TimeDC
The following command will then use the buffers we just generated to distill every dataset:
```
sh ./scipts_distill/weather.sh
```

Additional experimental results can be see at [Additional_experiment.md](./Additional_experiment.md)