# Shopee Challenge
This repo consists of modules used for shopee address data extraction challenge. 
We implemented NER extraction using bert(indo-nlu).
The input data must be in the IOB format to do training. The module includes scripts to generate 
in the IOB format.
# Main function
## Single parameter training
e.g. `python main.py --train_dataset_path XXX --valid_dataset_path YYY --test_dataset_path ZZZ --predict_only False --n_epochs 10 --validate_epoch 2 --device cuda --shopee_prediction_file AAA`

## Single parameter prediction
e.g. `python main.py --test_dataset_path ZZZ --predict_only True --device cuda --result_path BBB --convert_shopee True --shopee_prediction_file AAA`


# Reference
indo-nlu: https://github.com/indobenchmark/indonlu   
 
#### Reference notebook 
Training: notebooks/Shopee_ExecutorMain.ipynb   
Debug: notebooks/Test_executor.ipynb
