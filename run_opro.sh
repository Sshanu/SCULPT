#!/bin/bash
python src/opro/opro.py --task $1 --initial_prompt_type long_humanwritten --prompts prompts/$1/prompt_1.txt
