#!/bin/bash

python address_database.py   --dim 768 \
                        --num_chunk 3 \
                         --data_path "audio/address_audio" \
                         --additional_address "audio/additional_address" \
                         --save_path "data/address_new_db_chunk3.pt"
