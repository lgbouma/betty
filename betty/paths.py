import os
from betty import __path__

HOMEDIR = os.path.expanduser('~')

BETTYDIR = os.path.join(HOMEDIR, '.betty')
if not os.path.exists(BETTYDIR):
    print(21*'*-')
    print(f'WRN! Making a hidden cache directory at {BETTYDIR}')
    print(21*'*-')
    os.mkdir(BETTYDIR)

DATADIR = os.path.join(os.path.dirname(list(__path__)[0]), 'data')
TESTDATADIR = os.path.join(DATADIR, 'test_data')
TEXDATADIR = os.path.join(DATADIR, 'TeX')
RESULTSDIR = os.path.join(os.path.dirname(list(__path__)[0]), 'results')
TESTRESULTSDIR = os.path.join(RESULTSDIR, 'test_results')

dirlist = [DATADIR, TESTDATADIR, RESULTSDIR, TESTRESULTSDIR]
for d in dirlist:
    if not os.path.exists(d):
        print(21*'*-')
        print(f'WRN! Making {d}')
        print(21*'*-')
        os.mkdir(d)
