
# author: Quantium Rock
# license: MIT

from data_primary import PrimaryData
from data_secondary import SecondaryData
from data_tertiary import TertiaryData


def updateDB():

    primaryData = PrimaryData()
    secondaryData = SecondaryData()
    tertiaryData = TertiaryData()

    print('\n### PRIMARY DB ###')
    primaryData.checkDB()

    if primaryData.missing_years:
        # user confirmation
        input('\nUpdate primary database?\n> Press Enter to proceed\n\n>>> ')
        primaryData.updateDB()
    
    print('\n### SECONDARY DB ###')
    secondaryData.checkDB()
    print('\n> Updating secondary data...')
    secondaryData.updateDB()

    print('\n### TERTIARY DB ###')
    tertiaryData.checkDB()
    print('\n> Updating tertiary data...')
    tertiaryData.updateDB()


if __name__ == "__main__":
    updateDB()
