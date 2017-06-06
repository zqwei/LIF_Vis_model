#!/bin/bash

i=122
python syn_weight_amp.py $i
rm output_ll2_g8_8_test500ms_inh_lif_syn_z$i/*.*
qsub *z$i.qsub

while [ ! -f output_ll2_g8_8_test500ms_inh_lif_syn_z$i/tot_f_rate.dat ]; do
  echo output_ll2_g8_8_test500ms_inh_lif_syn_z$i
  sleep 2m
done

echo "qsub job done!"

python plot_spikes_file_idx.py $i
python plot_tot_firing_rate_comparison.py $i
