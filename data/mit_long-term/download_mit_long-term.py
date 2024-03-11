# -*- coding: utf-8 -*-
"""Script for formatting the MIT-Long-Term ECG Database

Steps:
    1. Download the ZIP database from https://physionet.org/content/ltdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-long-term-ecg-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import os

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

database_path = "./mit-bih-long-term-ecg-database-1.0.0/"

# Check if expected folder exists
if not os.path.exists(database_path):
    url = "https://physionet.org/static/published-projects/ltdb/mit-bih-long-term-ecg-database-1.0.0.zip"
    download_successful = nk.download_zip(url, database_path)
    if not download_successful:
        raise ValueError(
            "NeuroKit error: download of MIT-Arrhythmia database failed. "
            "Please download it manually from https://alpha.physionet.org/content/mitdb/1.0.0/ "
            "and unzip it in the same folder as this script."
        )

data_files = [database_path + file for file in os.listdir(database_path) if ".dat" in file]



dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))


    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 1]})
    data["Participant"] = "MIT-LongTerm_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 128
    data["Database"] = "MIT-LongTerm"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-LongTerm_%.2i" %(participant)
    anno["Sampling_Rate"] = 128
    anno["Database"] = "MIT-LongTerm"

    # Select only 2h of recording (otherwise it's too big)
    data = data[460800:460800*3].reset_index(drop=True)
    anno = anno[(anno["Rpeaks"] > 460800) & (anno["Rpeaks"] <= 460800*3)].reset_index(drop=True)
    anno["Rpeaks"] = anno["Rpeaks"] - 460800


    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)



# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)


# Quick test
#import neurokit2 as nk
#nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1001])
