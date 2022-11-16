# author: Quantium Rock
# license: MIT

from data_primary import PrimaryData
from data_secondary import SecondaryData
from data_tertiary import TertiaryData
from merge import merge_db_data
from volatility import VolatilityFeatures
from tendency import TendencyFeatures
from cot.cot_get_hist import get_cot

def updateDB():

    primaryData = PrimaryData()
    secondaryData = SecondaryData()
    tertiaryData = TertiaryData()

    dirs = ["primary", "secondary", "tertiary"]
    for d in dirs:
        primaryData.deleteFolder(d, "2022")
    primaryData.deleteFolder("merge")

    print("\n### PRIMARY DB ###")
    primaryData.checkDB()

    if primaryData.missing_years:
        # user confirmation
        # input("\nUpdate database?\n> Press Enter to proceed\n\n>>> ")
        primaryData.updateDB()

    print("\n### SECONDARY DB ###")
    secondaryData.checkDB()
    secondaryData.updateDB()

    print("\n### TERTIARY DB ###")
    tertiaryData.checkDB()
    tertiaryData.updateDB()

    print("\n### COMITMENT OF TRADERS DB ###")
    get_cot()

    print("\nPrimary, Secondary & Tertiary DB up to date! \nCOT dowloaded and up to date!")
    # input("\nDo you wanna merge db's?\n> Press Enter to proceed\n\n>>> ")

    print("\n### MERGE DB DATA ###")
    merge_db_data()

    print("\nDB's merged successfully.")
    # input("\nDo you update indicators?\n> Press Enter to proceed\n\n>>> ")

    print("\n### FEATURE ENGINEERING ###")
    TendencyFeatures().getTendency()
    VolatilityFeatures().getVolatility()

    print("\nYour DB is up to date. Bye!\n")


if __name__ == "__main__":
    updateDB()
