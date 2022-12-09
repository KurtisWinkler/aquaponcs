#!/bin/bash

test -e ssshtest || wget -q https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif'
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --min_distance 3
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --min_thresh_maxima 0.8
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --min_thresh_contours 0.8
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --thresh_step 3
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --init_filter [['area', 20, None]]
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --sim_filter [['perimeter', 0.2]]
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --out_filter [['axis_major_length', 0.2, 1]]
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --best_filter [['roughness_surface', 'min', 1]]
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --no_init_filter
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --no_sim_filter
assert_no_stdout
assert_exit_code 0

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif' --no_out_filter
assert_no_stdout
assert_exit_code 0