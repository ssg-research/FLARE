#!/bin/bash

###################################################################################################################
# The first batch of experiments check false negatives and positives without any evasion/robustness check
printf "The first batch of experiments check false negatives and positives without any evasion/robustness check"
printf "\n"

# The first loop iterates over games
# The second for loop iterates over victims
# The third for loop iterates over suspected agents
# The fourth loop iterates over model id numbers
for game in "Pong" "MsPacman"
do
    for victim in "a2c" "dqn" "ppo"
    do
        # Fingerprint generation
        python main.py --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --generate-fingerprint --eps 0.05 --cuda
        python main.py --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --eps 0.05 --suspected-agent-mode $victim \
                    --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --ver-slength 40 --cuda
        for suspected in "a2c" "dqn" "ppo" 
        do
            for (( id=1; id<=5; id++ ))
            do
                # Fingerprint verification
                python main.py  --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --eps 0.05 --suspected-agent-mode $suspected \
                                --suspected-agent-path output/$game/$suspected/fingerprint/model$id.pt --ver-slength 40 --cuda
            done
        done
    done
done 

# Print a newline at the end of the output
printf "\n"

###################################################################################################################
# The second batch of experiments check evasion of verification by returning random action values with a percentage
printf "The second batch of experiments check evasion of verification by returning random action values with a percentage"
printf "\n"

# The first loop iterates over games
# The second for loop iterates over victims
# The third for loop iterates over suspected agents
# The fourth loop iterates over model id numbers
for game in "Pong" "MsPacman" 
do
    for victim in "a2c" "dqn" "ppo"
    do
        for end in 100 200 300 400
        do
            for ratio in 0.0 0.2 0.4 0.6 0.8 1.0
            do
                python main.py --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --eps 0.05 --suspected-agent-mode $victim \
                            --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --ver-slength $end --random-action-ratio $ratio 
            done
        done
    done
done

# Print a newline at the end of the output
printf "\n"

###################################################################################################################
# The third batch of experiments check robustness of fingerprints using finetuning
printf "The third batch of experiments check robustness of fingerprints using finetuning"
printf "\n"

# The first loop iterates over games
# The second for loop iterates over victims
# The third for loop iterates over finetuned agents
for game in "Pong" "MsPacman" 
do
    for victim in "a2c" "dqn" "ppo"
    do
        python main.py --game-mode finetune --env-name $game --victim-agent-mode $victim
        for (( id=0; id<=200; id+=50 ))
        do
            python main.py --game-mode test --env-name $game --suspected-agent-mode $victim --suspected-agent-path output/$game/$victim/finetune/finetuned_model$id.pt
            python main.py --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --eps 0.05 --suspected-agent-mode $victim \
                    --suspected-agent-path output/$game/$victim/finetune/finetuned_model$id.pt --ver-slength 40
            rm output/$game/$victim/finetune/finetuned_model$id.pt
        done
    done
done

# Print a newline at the end of the output
printf "\n"

###################################################################################################################
# The fourth batch of experiments check robustness of fingerprints using finepruning
printf "The fourth batch of experiments check robustness of fingerprints using finepruning"
printf "\n"

# The first loop iterates over games
# The second for loop iterates over victims
# The third loop iterates over the pruning level
# The fourth loop iterates over model id numbers
for game in "MsPacman" "Pong" 
do
    for victim in "a2c" "dqn" "ppo"
    do
        python main.py --game-mode fineprune --env-name $game --victim-agent-mode $victim --suspected-agent-mode $victim
        for rate in 25 50 75 90
        do
            for id in 0 200
            do
                python main.py --game-mode test --env-name $game --suspected-agent-mode $victim --suspected-agent-path output/$game/$victim/fineprune/pruned_model$id-prate-$rate.pt
                python main.py --game-mode fingerprint --env-name $game --adversary cosfwu --victim-agent-mode $victim --eps 0.05 --suspected-agent-mode $victim \
                        --suspected-agent-path output/$game/$victim/fineprune/pruned_model$id-prate-$rate.pt --ver-slength 40
                rm output/$game/$victim/fineprune/pruned_model$id-prate-$rate.pt
            done
        done
    done
done
# Print a newline at the end of the output
printf "\n"

###################################################################################################################
# The fifth batch of experiments check evasion of verification using adversarial example detection and action recovery
printf "The fifth batch of experiments check evasion of verification using adversarial example detection and action recovery"
printf "\n"

# The first loop iterates over games
# The second for loop iterates over victims
for game in "MsPacman" "Pong" 
do
    for victim in "a2c" "dqn" "ppo"
    do
        python main.py --game-mode test --env-name $game --victim-agent-mode $victim --suspected-agent-mode $victim --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --vf1
        python main.py --game-mode fingerprint --env-name $game --victim-agent-mode $victim --suspected-agent-mode $victim \
                    --adversary cosfwu --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --vf1
        for ratio in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            python main.py --game-mode test --env-name $game --victim-agent-mode $victim --suspected-agent-mode $victim \
                        --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --vf2 --vf2-random-action-ratio $ratio
            python main.py --game-mode fingerprint --env-name $game --victim-agent-mode $victim --suspected-agent-mode $victim \
                         --adversary cosfwu --suspected-agent-path output/$game/$victim/fingerprint/model_original.pt --vf2 --vf2-random-action-ratio $ratio 
        done
    done
done


# Print a newline at the end of the output
printf "\n"