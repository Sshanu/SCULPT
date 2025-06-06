#!/bin/bash
python src/protegi/main.py --task $1 --prompt_length long --prompts prompts/$1/prompt_1.txt --rounds 6
