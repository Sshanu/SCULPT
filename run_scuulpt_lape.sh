#!/bin/bash

python src/sculpt/main.py --task $1 --prompt_length long --prompts prompts/$1/lape.txt --aggregate_feedbacks explicit --run_name Lape_Sculpt --rounds 8