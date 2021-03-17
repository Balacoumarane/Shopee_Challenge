import click
from NER_Module import execute_main, get_logger

logger = get_logger(__name__)


@click.command()
@click.option('--train_dataset_path', default=None, type=str)
@click.option('--valid_dataset_path', default=None, type=str)
@click.option('--test_dataset_path', default=None, type=str)
@click.option('--model_dir', default='models', type=str, required=True)
@click.option('--model_filename', default=None, type=str)
@click.option('--predict_only', default=False, type=bool)
@click.option('--n_epochs', default=10, type=3)
@click.option('--exp_id', default=None, type=str)
@click.option('--random_state', default=33, type=int)
@click.option('--device', default='cuda', type=str)
@click.option('--validate_epoch', default=1, type=int)
@click.option('--early_stop', default=3, type=int)
@click.option('--criteria', default=None, type=str)
@click.option('--result_path', default=None, type=str)
@click.option('--result_filename', default='result', type=str, required=True)
def main(train_dataset_path, valid_dataset_path, test_dataset_path, model_dir, model_filename, predict_only, n_epochs,
         exp_id, random_state, device, validate_epoch, early_stop, criteria, result_path, result_filename):

    execute_main(train_dataset_path=train_dataset_path, valid_dataset_path=valid_dataset_path,
                 test_dataset_path=test_dataset_path, model_dir=model_dir, model_filename=model_filename,
                 predict_only=predict_only, n_epochs=n_epochs, exp_id=exp_id, random_state=random_state, device=device,
                 validate_epoch=validate_epoch, early_stop=early_stop, criteria=criteria,
                 result_path=result_path, result_filename=result_filename)


if __name__ == '__main__':
    main()
