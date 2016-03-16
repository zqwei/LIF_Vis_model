# cd LGN/
python generate-LGN-rates.py
python generate-LGN-spikes.py

cd ../tw_data/ll2_tw_build/
python run_traveling_waves.py
# This seems duplicated
# python run_traveling_waves.py