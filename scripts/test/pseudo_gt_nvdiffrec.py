import os
os.environ['IMAEGINT_PSEUDO_GT'] = '1'
from orb.utils.test import compute_metrics


if __name__ == "__main__":
    compute_metrics('nvdiffrec')
