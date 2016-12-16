import numpy as np


def read_lines(file_name):
    """
    read lines from file
    :param file_name: file_name
    :return: data line list
    """
    file_object = open(file_name, 'r').readlines()
    return file_object


def read_lines_svm_data_format(file_name):
    """
    read lines from file
    :param file_name: file_name
    :return: data line list
    """
    output = []
    file_object = open(file_name, 'r')
    for lines in file_object.readlines():
        list_int = map(int, lines.split("\t"))
        output.append(np.array(list_int))
        # output.append(list_int)

    return np.array(output)
    # return  output
