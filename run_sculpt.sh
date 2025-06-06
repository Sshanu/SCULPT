#!/bin/bash

python src/sculpt/main.py --task $1 --prompt_length long --prompts prompts/$1/prompt_1.txt --aggregate_feedbacks explicit --run_name Sculpt --rounds 8