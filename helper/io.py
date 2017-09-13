"""
=========
IO Helper
=========

This module contains JSON and CSV file readers,
which reads data inside them and return a Python
`dict` object.

"""

def json2dict(filepath: str) -> dict:
    """JSON to dict
    """
    import json
    with open(filepath, encoding='utf-8') as json_file:
        return json.load(json_file)

def csv2list(filepath: str) -> list:
    """CSV to dict
    All numerical values in csv without a
    decimal point will be evaluated as `int`,
    otherwise as `float`.
    """
    import csv
    l = []
    with open(filepath, encoding='utf-8') as csv_file:
        dr = csv.DictReader(csv_file)
        for row in dr:
            l.append(
                dict(
                    [
                        # if no decimal point presents,
                        # evaluate it as an integer
                        (k, int(v))
                        if v and '.' not in v
                        else
                        (k, float(v))
                        if v
                        else
                        # if 
                        (k, None)
                        for k, v in row.items()
                    ]
                )
            )
    return l

def show_io_helper_doc():
    print('Running IO Helper directly is not supported')

if __name__ == '__main__':
    show_io_helper_doc()
