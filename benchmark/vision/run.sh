export CUDA_VISIBLE_DEVICES=1
python exp_mem_speed.py --mode linear_scan 
exit
python exp_mem_speed.py --mode binary_search_max_width
python exp_mem_speed.py --mode binary_search_max_layer
python exp_mem_speed.py --mode binary_search_max_input_size
