"""
Built-in datasets for demonstration, educational and test purposes.
"""


def get_dataset(d):
    import pandas
    import os

    return pandas.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "package_data",
            "datasets",
            d,
        )
    )
