import shutil
import os
import random
import argparse
import warnings
from distutils.dir_util import copy_tree


def cat_to_dataset(datapath="", percentage: list = [.7, .2, .1], seed=3):
    '''
    Divides a directory containing only contain folders with each category to
    classify into 'train', 'valid' and 'test' folders contain each category for
    ML with custom splitting percentages percentage[0] +
    percentage[1] + percentage[2] should be = 1 

    Parameters
    ----------
    datapath : string
         Datapath directory should only contain folders with each category to
         classify. Each category fodler must only contain image files of that
         category. The default is "".
    percentage[0] : float
        Training set percentage (percentage[0] + percentage[1] +
        percentage[2] should be = 1). The default is 0.7.
    percentage[1] : float
        Validation set percentage (percentage[0] + percentage[1] +
        percentage[2] should be = 1). The default is 0.2.
    percentage[2] : float
        Test set percentage (percentage[0] + percentage[1] +
        percentage[2] should be = 1). The default is 0.1.
    seed : integer
        Random seed. The default is 3.

    Returns
    -------
    None.

    '''

    dirs = os.listdir(datapath)

    for cat_name in dirs:
        cat_path = datapath + '/' + cat_name + '/'
        os.makedirs(datapath + '/train/', exist_ok=True)
        os.makedirs(datapath + '/valid/', exist_ok=True)
        os.makedirs(datapath + '/test/', exist_ok=True)
        os.makedirs(datapath + '/train/' + cat_name + '/', exist_ok=True)
        os.makedirs(datapath + '/valid/' + cat_name + '/', exist_ok=True)
        os.makedirs(datapath + '/test/' + cat_name + '/', exist_ok=True)
        cat_files = os.listdir(cat_path)
        random.Random(seed).shuffle(cat_files)  # to shuffle category data
        cat_total_files = len(cat_files)
        cat_trainset_max = round(float(percentage[0]) * cat_total_files)
        cat_validset_max = round(percentage[1] * cat_total_files)
        cat_trainset = cat_files[0:cat_trainset_max]
        cat_validset = cat_files[cat_trainset_max:
                                 cat_trainset_max + cat_validset_max]
        cat_testset = cat_files[cat_trainset_max + cat_validset_max:]
        for files in cat_trainset:
            shutil.move(cat_path + files, datapath +
                        '/train/' + cat_name + '/')
        for files in cat_validset:
            shutil.move(cat_path + files, datapath +
                        '/valid/' + cat_name + '/')
        for files in cat_testset:
            shutil.move(cat_path + files, datapath + '/test/' + cat_name + '/')
        os.rmdir(cat_path)


def dataset_to_cat(datapath=""):
    '''
    Groups a directory containing a 'train', 'test' and 'valid' (code is case
    sensitive) folders for ML with category sub folders into category folders.
    Parameters
    ----------
    datapath : string
        Datapath directory should only contain a 'train', 'test' and 'valid'
        (code is case sensitive) folders. Each fodler should only contain
        folders with each category to classify. Each category fodler must
        only contain image files of that category. The default is "".
    Returns
    -------
    None.
    '''

    dirs = os.listdir(datapath + '/train/')

    for cat_name in dirs:

        os.makedirs(datapath + '/' + cat_name + '/', exist_ok=True)
        dirs_train = os.listdir(datapath + '/train/' + cat_name + '/')
        dirs_valid = os.listdir(datapath + '/valid/' + cat_name + '/')
        dirs_test = os.listdir(datapath + '/test/' + cat_name + '/')

        for files in dirs_train:
            shutil.move(datapath + '/train/' + cat_name + '/' +
                        files, datapath + '/' + cat_name + '/')
        for files in dirs_valid:
            shutil.move(datapath + '/valid/' + cat_name + '/' +
                        files, datapath + '/' + cat_name + '/')
        for files in dirs_test:
            shutil.move(datapath + '/test/' + cat_name + '/' +
                        files, datapath + '/' + cat_name + '/')

    shutil.rmtree(datapath + '/train/', ignore_errors=True)
    shutil.rmtree(datapath + '/valid/', ignore_errors=True)
    shutil.rmtree(datapath + '/test/', ignore_errors=True)


def main(defined_action_map={}):

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("input", action='store', help='Folder containing the dataset.')
    parser.add_argument("output", action='store', help='Path to new segmented dataset.')

    parser.add_argument('--percentages', action='store',
                        help='Percentage of dataset to [trainset,validset,testset]. Default is [0.7,0.2,0.1].')
    parser.add_argument('--seed', action='store',
                        help='Random seed. Default is 3.')
    parser.set_defaults(percentages=[0.7, 0.2, 0.1])
    parser.set_defaults(seed=3)
    args = parser.parse_args()

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.mkdir(args.output)

    copy_tree(args.input, args.output)
    path = args.output
    perc = str(args.percentages).strip('][').split(',')
    print(perc)
    percent = [float(x) for x in perc]

    cat_to_dataset(path, percentage=percent,  seed=args.seed)


if __name__ == '__main__':
    main()
