
# author: Quantium Rock
# license: MIT

from sys import argv
from data import PrimaryData

if __name__ == "__main__":

	if len(argv) > 1:
		if 'check' in argv[1]:
			PrimaryData().checkDB()
		elif 'update' in argv[1]:
			PrimaryData().updateDB()
		elif 'mids' in argv[1]:
			pass
		else:
			print('-h for help')
	else:
		print('HELP')
