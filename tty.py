#!/usr/bin/env python3

import os
from sys import argv

if len(argv) != 2:
    raise ValueError("missing arguments")

st = os.stat(argv[1], follow_symlinks=True)
print("file[%x:%x]" % (0, st.st_ino))
print("tty[%x:%x]" % (st.st_rdev, st.st_dev))
