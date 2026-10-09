"""Read an archived receipt in either its original or losslessly gzipped form."""
import gzip
import json
from pathlib import Path


def read_receipt(path):
    path=Path(path)
    if not path.exists():path=path.with_name(path.name+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:return json.load(stream)
