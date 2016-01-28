# get cell models

import csv
from allensdk.api.queries.biophysical_perisomatic_api import BiophysicalPerisomaticApi

bp = BiophysicalPerisomaticApi('http://api.brain-map.org')
bp.cache_stimulus = False # change to True to download the large stimulus NWB file

with open("V1_L4_antona.csv") as csvfile, open ("V1_L4.csv", 'w') as writefile:
    reader = csv.reader(csvfile, delimiter = ' ')
    writer = csv.writer(writefile, delimiter = ' ')
    for row in reader:
        if reader.line_num > 1:
            morphologyFileName = row[6]
            cell_parFileName = row[7]

            try:
                cutFileName = morphologyFileName.rindex('/')
                morphologyFileName = morphologyFileName[(cutFileName+1):]
                morphologyFileName = 'neuronal_model/' + morphologyFileName
            except ValueError:
                pass

            try:
                cutFileName = cell_parFileName.rindex('/')
                cell_parFileName = cell_parFileName[(cutFileName+1):]
                neuronal_model_id = [int(s) for s in cell_parFileName.split('_') if s.isdigit()]
                cell_parFileName = 'neuronal_model/' + cell_parFileName
                bp.cache_data(neuronal_model_id[0], working_directory='neuronal_model')
            except ValueError:
                pass

            row[6] = morphologyFileName
            row[7] = cell_parFileName

        writer.writerows([row])
