# from allensdk.api.queries.glif_api import GlifApi
# from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.core.cell_types_cache import CellTypesCache

import numpy as np
import pandas as pd


ctc = CellTypesCache()
features = ctc.get_ephys_features()
tauR2CUnitRatio = 1000.0



cellName = {318808427: 'Nr5a1', 395830185: 'Scnna1', 314804042: 'Rorb',
			330080937: 'Pvalb', 318331342: 'Pvalb'}

cellNickName = {318808427: 'Nr5a1', 395830185: 'Scnn1a', 314804042: 'Rorb',
				330080937: 'PV1', 318331342: 'PV2'}

ephysData = pd.read_csv('ephys_features.csv', index_col=0)

lenCellType = len(cellName)

LIFModelDF = np.ndarray((lenCellType+2,),
                     dtype=[('type', '|S10'), ('tau_m', np.float64),
                            ('C_m', np.float64), ('E_L', np.float64),
                            ('V_th', np.float64), ('V_reset', np.float64),
                            ('t_ref', np.float64)])

LIFModelDF[0] = ('LIF_exc', 7.5, 250.0, -70.0, -55.0, -70.0, 3.0)
LIFModelDF[1] = ('LIF_ihn', 100.0, 250.0, -70.0, -55.0, -70.0, 3.0)

for nCell in range(lenCellType):
    nCell_type = cellNickName.values()[nCell]
    nCell_tau_m = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'tau'].values
    nCell_R = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'ri'].values
    nCell_E_L = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'vrest'].values
    # nCell_V_th = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'threshold_v_long_square'].values
    # nCell_V_reset = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'trough_v_long_square'].values
    nCell_V_th = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'threshold_v_short_square'].values
    nCell_V_reset = ephysData.loc[ephysData['specimen_id'] == cellNickName.keys()[nCell], 'trough_v_short_square'].values
    nCell_t_ref = 3.0
    LIFModelDF[nCell+2] = (nCell_type, nCell_tau_m, nCell_tau_m/nCell_R*tauR2CUnitRatio,
    						nCell_E_L, nCell_V_th, nCell_V_reset, nCell_t_ref)

print(LIFModelDF)

np.save('LIFModel.npy', LIFModelDF)
