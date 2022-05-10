# >>> conda initialize >>>
__conda_setup="$('/vol/research/dcase2022/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/vol/research/dcase2022/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/vol/research/dcase2022/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/vol/research/dcase2022/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate py38
######################## ENVIRONMENT ########################
which python
cd /vol/research/dcase2022/project/byol-a

######################## SETUP ########################
EXP_NAME="exp16_byol_model"

######################## RUNNING ENTRY ########################
# $1=pcen@delta_mfcc

python3 train.py /vol/research/dcase2022/datasets/AudioSet_SED/data