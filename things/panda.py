

import pandas as pd 


def run():
	data = {'nombre':['ovidio', 'hidalgo', 'samuel'],
			'edad':[21,21,12],
			'pais':['COL', 'COL', 'COL']}

	dframe = pd.DataFrame(data=data)
	print(dframe)
	print("="*40)
	print(dframe[['pais']])
	print("="*40)
	print(dframe.columns)

if __name__ == '__main__':
	run()
