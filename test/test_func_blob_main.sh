#!/bin/bash

test -e ssshtest || wget -q https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run test_blob_main python ../blob_main.py --file_name '../example_images/ex1.tif'
assert_no_stdout
assert_exit_code 0