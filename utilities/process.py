import pandas as pd
import tensorflow as tf
import re
import os
import json
from tensorflow.keras.layers import TextVectorization
from utilities.utils import *
from utilities.plot import *

with open('../hyperparameters.json','r') as f:
    data=json.load(f)