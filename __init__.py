from __future__ import absolute_import
import os, sys

dossier = os.path.dirname(os.path.abspath(__file__))

while not dossier.endswith('Blabladata'):
    dossier = os.path.dirname(dossier)

dossier = os.path.dirname(dossier)

if dossier not in sys.path:
    sys.path.append(dossier)
print sys.path

import preprocessing
