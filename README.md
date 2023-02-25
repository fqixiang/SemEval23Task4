# Epicurus at SemEval-2023 Task 4

## Project description
This repository is created for reproducing the approach and results 
in the following paper: 
[Epicurus at SemEval-2023 Task 4: Improving Prediction of Human Values from Arguments by Leveraging Their Definitions](insert url).

## Set up environment
```
conda create --name epicurus python=3.10 pip
conda activate epicurus
pip install -r requirements.txt
cd ./src
```

## Download data 
Please download the challenge data from [this link](https://zenodo.org/record/7550385#.Y_kO--zMJqs) to the folder ./data/raw/touche23

## Model training
```
python train.py --batch_size 32 --gradient_step_size 4 --definition description --weighted_loss 'weighted' --test_mode no
python train.py --batch_size 32 --gradient_step_size 4 --definition description --weighted_loss 'not_weighted' --test_mode no
python train.py --batch_size 32 --gradient_step_size 4 --definition survey --weighted_loss 'weighted' --test_mode no
python train.py --batch_size 32 --gradient_step_size 4 --definition survey --weighted_loss 'not_weighted' --test_mode no
```


## Get predictions
```
python predict.py --definition description --weighted_loss weighted --model_number [model number] --test_mode no
python predict.py --definition description --weighted_loss not_weighted --model_number [model number] --test_mode no
python predict.py --definition survey --weighted_loss weighted --model_number [model number] --test_mode no
python predict.py --definition survey --weighted_loss not_weighted --model_number [model number] --test_mode no
```

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)
