
# DOWNLOAD FULL COMMITMENT OF TRADERS REPORTS FROM CFTC

# author: Quantium Rock
# license: MIT
# date: August 2022


"""
The Commodity Futures Trading Commission (US Derivatives Trading Commission or CFTC) publishes the Commitments of Traders (COT) reports to help the public understand market dynamics. Specifically, the COT reports provide a breakdown of each Tuesday’s open interest for futures and options on futures markets in which 20 or more traders hold positions equal to or above the reporting levels established by the CFTC.

The COT reports are based on position data supplied by reporting firms (FCMs, clearing members, foreign brokers and exchanges). While the position data is supplied by reporting firms, the actual trader category or classification is based on the predominant business purpose self-reported by traders on the CFTC Form 40 and is subject to review by CFTC staff for reasonableness. CFTC staff does not know specific reasons for traders’ positions and hence this information does not factor in determining trader classifications. In practice this means, for example, that the position data for a trader classified in the “producer/merchant/processor/user” category for a particular commodity will include all of its positions in that commodity, regardless of whether the position is for hedging or speculation. Note that traders are able to report business purpose by commodity and, therefore, can have different classifications in the COT reports for different commodities. For one of the reports, Traders in Financial Futures, traders are classified in the same category for all commodities.

Due to legal restraints (CEA Section 8 data and confidential business practices), the CFTC does not publish information on how individual traders are classified in the COT reports.

Generally, the data in the COT reports is from Tuesday and released Friday. The CFTC receives the data from the reporting firms on Wednesday morning and then corrects and verifies the data for release by Friday afternoon.
"""

import os
import requests

from bs4 import BeautifulSoup
from pathlib import Path
from zipfile import ZipFile


def cot_bulk_downloader():
    """ Downloads all standard Commitment of Traders Futures only reports. """
    url = 'https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm'
    out_path = Path.cwd() / 'cots_raw'
    if out_path.exists():  # avoid re-downloading the script if the files have already been downloaded
        return 0
    os.makedirs(out_path, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for name in soup.findAll('a', href=True):
        file_url = name['href']
        if file_url.endswith('.zip'):
            filename = file_url.split('/')[-1]
            if 'deacot' in filename:
                out_name = out_path.joinpath(filename)
                zip_url = 'https://www.cftc.gov/files/dea/history/' + filename
                req = requests.get(zip_url, stream=True)
                if req.status_code == requests.codes.ok:
                    if filename.__contains__('2003'):
                        break
                    print(f'Downloading {filename} . . .')
                    with open(out_name, 'wb') as f:
                        for chunk in req.iter_content():
                            if chunk:  # ignore keep-alive requests (?)
                                f.write(chunk)
                        f.close()

    print('Finished downloading reports')
    return 0

def bulk_unzipper(directory):
    """ Unzips all archives within the CoT directory. """
    for file in directory.iterdir():
        if str(file).endswith('.zip'):
            print(f'Unzipping {file} . . .')
            with ZipFile(file, 'r') as zip_object:
                filename = zip_object.filename.split('/')[-1]  # after split consider only the 'deacotXXXX.zip part
                filename = filename.split('.zip')[0]           # consider only the 'deacotXXXX' string
                zip_object.extractall(directory / filename)
    return 0

if __name__ == '__main__':
    try:
        return_val = cot_bulk_downloader()
        if return_val == 0:  # hunky-dory so proceed to unzip
            cot_dir = Path.cwd() / 'cots_raw'  # path created by bulk downloader so no I/O error 
            bulk_unzipper(cot_dir)
    except Exception as exception:
        print(f'Unexpected error: {exception}')

### END