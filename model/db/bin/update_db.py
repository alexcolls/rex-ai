
# author: Quantium Rock
# license: MIT

import json
from db.bin.primary import PrimaryData
from db.bin.secondary import SecondaryData


def updateDB():

    with open("env/variables.json") as x:
        FIRST_YEAR = json.load(x)['FIRST_YEAR']

    primaryData = PrimaryData( start_year=FIRST_YEAR )
    secondaryData = SecondaryData()

    print('\n### PRIMARY DB ###')
    primaryData.checkDB()

    if primaryData.missing_weeks or primaryData.missing_years:

        # user confirmation
        input('\nUpdate primary database?\n> Press Enter to proceed\n\n>>> ')

        # update DB    
        primaryData.updateDB()
    
    print('\n### SECONDARY DB ###')
    secondaryData.checkDB()
    secondaryData.updateDB()


if __name__ == "__main__":
    updateDB()

