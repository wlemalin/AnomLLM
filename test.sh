#!/bin/bash

data=("trend" "range" "point" "freq")
variants=("1shot-vision" "0shot-vision" "1shot-text" "0shot-text")

models=("gpt-4o" "gpt-4o-mini")

python src/batch_api.py --data trend --model gpt-4o --variant 1shot-vision
for model in "${models[@]}"; do
  for datum in "${data[@]}"; do
    for variant in "${variants[@]}"; do
      session_name="${datum}_${model}_${variant}"
      command="python src/batch_api.py --data $datum --model $model --variant $variant"
      tmux new-session -d -s "$session_name" "$command"
    done
  done
done

# models=("internvlm-76b")

# # python src/online_api.py --data trend --model internvlm-76b --variant 1shot-vision
# for model in "${models[@]}"; do
#   for datum in "${data[@]}"; do
#     for variant in "${variants[@]}"; do
#       session_name="${datum}_${model}_${variant}"
#       # Kill the existing session if it exists
#       tmux has-session -t "$session_name" 2>/dev/null
#       if [ $? -eq 0 ]; then
#         tmux kill-session -t "$session_name"
#       fi
#       command="python src/online_api.py --data $datum --model $model --variant $variant"
#       tmux new-session -d -s "$session_name" "$command"
#     done
#   done
# done
