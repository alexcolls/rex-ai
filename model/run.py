
### MAIN RUNNER

import os
import json
from db.bin.update_db import updateDB

if __name__ == "__main__":

    with open('env/variables.json') as d:
        variables = json.load(d)
        FIRST_RUN = variables['FIRST_RUN']

    if FIRST_RUN:

        from src.first_run import firstRun
        firstRun()    

    else:

        updateDB()

        ans = input('\n> Do you want to open the model dashboard? \n\n >>> Press Enter \n ')

        os.system("python src/dashboards/dash_market.py")




    




