import pandas as pd
import numpy as np
import os

def tot_f_rate(spk_f_name, out_f_name, in_t_omit, t_tot_m_post_t_omit, t_tot, N_cells):
  # Create a data frame that would contain entries for all cells (and not only non-zero ones that we get from the spk file).
  all_df = pd.DataFrame(np.zeros(N_cells), columns=['rates_t_omit'])
  all_df['rates_tot'] = np.zeros(N_cells)

  if (os.stat(spk_f_name).st_size != 0):
    df = pd.read_csv(spk_f_name, header=None, sep=' ')
    df.columns = ['t', 'gid']
    rates = df.groupby('gid').count() * 1000.0 / t_tot # Time is in ms and rate is in Hz.
    rates.columns = ['rates_tot']

    df1 = df[df['t'] >= in_t_omit]
    df1 = df1[df1['t'] <= t_tot_m_post_t_omit]
    rates1 = df1.groupby('gid').count() * 1000.0 / (t_tot_m_post_t_omit - in_t_omit) # Time is in ms and rate is in Hz.
    rates1.columns = ['rates_t_omit']
    rates1['rates_tot'] = rates['rates_tot'] # Add the new column to rates1.

    # The important point here is that we use gid as indices from 0 to N_cells.  This occurs in "rates" because we
    # use "groupby" above.
    for gid in rates1.index:
      all_df['rates_t_omit'].loc[gid] = rates1['rates_t_omit'].loc[gid]
      all_df['rates_tot'].loc[gid] = rates1['rates_tot'].loc[gid]

  all_df.to_csv(out_f_name, header=None, sep=' ')

