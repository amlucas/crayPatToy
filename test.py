#! /usr/bin/env python

import sys
sys.path.insert(0, "build")
import libcptoy as cpt

size = 10000
c = cpt.CPToy(size)

c.cuda_test(1.0)
c.mpi_test(3.0)

