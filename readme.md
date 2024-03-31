# Less is More: Efficient Time Series Dataset Condensation via Two-fold Modal Matching


## Requirements
```
python >= 3.8
Pytorch >= 1.11
Numpy
Pandas
```

## Data Preparation
TimeDC is implemented on several public time series datasets.

- **Weather**, **Traffic**, **Electricity** and **ETT** from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

## Generating Expert Trajectories
Run each script in ```./scripts_buffer/``` to generate expert trajectories, for example
```
sh ./scripts_buffer/weather.sh
```

## Time Series Dataset Condensation with TimeDC
After obtaining expert trajectories, run each script in ```./scripts_distill/``` to perform time series dataset condensation, for example
```
sh ./scipts_distill/weather.sh
```

Additional experimental results can be see at [Additional_experiment.md](./Additional_experiment.md)
