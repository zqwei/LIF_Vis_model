for /l %%G in (101, 1, 999) do (
python make_run_file.py %%G
run_compile.bat)