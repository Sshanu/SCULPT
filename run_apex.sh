#!/bin/bash
python src/apex/main.py --task $1 --prompt_length long --prompts prompts/$1/prompt_1.txt --rounds 50
