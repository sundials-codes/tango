#!/bin/bash
python3 altered_tango_example.py | nl > tang.txt
python3 kinsol_simple.py | nl > outkin.txt
paste -d '\n' tang.txt outkin.txt | less
