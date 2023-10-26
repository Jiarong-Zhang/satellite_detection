import pandas as pd

import shutil

from tqdm import tqdm

data = pd.read_csv("test.csv")

filenames = data['filename'].tolist()

for i in tqdm(range(len(filenames))):

	#print(filenames)
	""" move file to test img folder"""
	shutil.move('train_img/'+filenames[i], 'test_img/'+filenames[i])
