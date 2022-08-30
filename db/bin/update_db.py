
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
        input('\nUpdate database?\n> Press Enter to proceed\n\n>>> ')
        primaryData.updateDB()
    
    print('\n### SECONDARY DB ###')
    secondaryData.checkDB()
    secondaryData.updateDB()

    print('\n### TERTIARY DB ###')
    tertiaryData.checkDB()
    tertiaryData.updateDB()


if __name__ == "__main__":
    updateDB()
