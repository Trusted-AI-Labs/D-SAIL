import shutil
import os
import random
import numpy as np
import argparse


def hospital_split(input_path, output_path, split_percentages=[0.5, 0.3, 0.2], seed=3):
    '''

    Parameters
    ----------
    number_of_datasplits : integer
        Number of data splits (hospitals needed). The default is 3.
    split_percentages : array
        split percentage by dataset (hospital) the number of components must be equal to 'number_of_datasplits'. Be careful that the sum of them is exactly = 1. The default is np.array([0.5,0.3,0.2]).
    seed : integer
        Random seed used. The default is 3.
    datapath : string
        Datapath directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category. The default is "".

    Returns
    -------
    None.

    '''

    for i, percent in enumerate(split_percentages):
        os.makedirs(output_path + '/H' + str(i+1)+'/')
        for cat_name in os.listdir(input_path):
            os.makedirs(output_path + '/H' + str(i+1)+'/'+cat_name + '/')
            for case in os.listdir(input_path + '/' + cat_name + '/'):
                os.makedirs(output_path + '/H' + str(i+1)+'/' +
                            cat_name + '/' + case + '/')

                cat_files = os.listdir(
                    input_path + '/' + cat_name + '/'+case+'/')
                cat_set = cat_files[int(sum(split_percentages[0:i])*len(cat_files)): int(
                    (sum(split_percentages[0:i])+split_percentages[i])*len(cat_files))]

                for file in cat_set:
                    shutil.copyfile(input_path + '/' + cat_name + '/'+case +
                                    '/'+file,
                                    output_path + '/H' + str(i+1)+'/' +
                                    cat_name + '/' + case + '/'+file)

        # cat_set_min = 0
        # cat_path = datapath + '/' + cat_name + '/'
        # cat_files = os.listdir(cat_path)
        # # random.Random(seed).shuffle(cat_files)  # to shuffle category data

        # for i, percent in enumerate(split_percentages):
        #     for case in os.listdir(cat_path):
        #         os.makedirs(datapath + '/H' + str(i+1)+'/' +
        #                     cat_name + '/' + case + '/')
        #         cat_set_max = int(round(percent * len(cat_files)))
        #         cat_set = cat_files[cat_set_min: cat_set_min + cat_set_max]
        #         cat_set_min += cat_set_max
        #         for file in cat_set:
        #             os.replace(cat_path + file, datapath + '/H' +
        #                         str(i+1)+'/' + cat_name + '/' + case + '/'+file)
        # shutil.rmtree(cat_path)


def main():

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        'input', help='directory should only contain folders with each category to classify. Each category fodler must only contain image files of that category.')
    parser.add_argument('output', help='Path to output folder')

    parser.add_argument('--percentages', action='store',
                        help='Percentage of dataset to each hospital. Default is [0.7,0.2,0.1].')
    parser.add_argument('--seed', action='store',
                        help='Random seed. Default is 3.')
    parser.set_defaults(percentages=[0.7, 0.2, 0.1])
    parser.set_defaults(seed=3)
    args = parser.parse_args()

    perc = str(args.percentages).strip('][').split(',')
    percent = [float(x) for x in perc]

    output_path = args.output+'Split_' + \
        str(int(percent[0]*100)) + '_' + \
        str(int(percent[1]*100))+'_'+str(int(percent[2]*100))

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    hospital_split(args.input, output_path,
                   split_percentages=percent, seed=args.seed)


if __name__ == '__main__':
    main()
