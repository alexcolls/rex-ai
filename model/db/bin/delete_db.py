
# author: Quantium Rock
# license: MIT

import os, shutil

def deleteDB():

    input('\n >  WARNING! Are you sure you want to delete all your database? \n\n > Press Enter to continue or Ctrl-C to exit. \n\n >>>')

    db_path = 'db/data/'
    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if os.path.isdir(folder_path):
            for year in os.listdir(folder_path):
                year_path = os.path.join(folder_path, year)
                if os.path.isdir(year_path):
                    try:
                        shutil.rmtree(year_path)
                    except Exception as e:
                        print('\n > Failed to delete %s. Reason: %s' % (folder_path, e))
            print('\n > %s database deleted succesfully!' % folder)
                
    print('\n')

if __name__ == "__main__":
    deleteDB()