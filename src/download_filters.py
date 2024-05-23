import re

from dotenv import dotenv_values

config = dotenv_values('.env')


def lst_filter(uri: str, product: str, version: str) -> bool:
    return (
        re.search(fr'{product}\.{version}/\d{{4}}\.\d{{2}}\.\d{{2}}/', uri) is not None
    )


def hdf_filter(uri: str) -> bool:
    return uri.endswith('.hdf')


