#!/usr/bin/env bash

CSJDATATOP=/DATA/CSJ_RAW/
CSJVER=usb                          # see kaldi/egs/csj/s5/run.sh about the version
CSJ_ROOT=$(pwd)/data
OUTPUT_PATH=${CSJ_ROOT}/csj-lhotse
set -eou pipefail

[[ $(uname) == 'Darwin' ]] && nj=$(sysctl -n machdep.cpu.thread_count) || nj=$(grep -c ^processor /proc/cpuinfo)

# TODO: move to lhotse.recipes.csj
echo "$0: Data Preparation"
local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} data/csj-data ${CSJVER}
local/csj_data_prep.sh data/csj-data
for eval_num in eval1 eval2 eval3 ; do
    local/csj_eval_data_prep.sh data/csj-data/eval ${eval_num}
done
for x in train eval1 eval2 eval3; do
    local/csj_rm_tag_sp_space.sh data/${x}
done

echo "$0: Prepare audio and supervision manifests for csj."
lhotse prepare csj $speech_root $transcripts_dir $output_dir

# TODO: lhotse feat extractを使う。
for x in train eval1 eval2 eval3; do
    mkdir -p ${data}/${feature_type}/${x}
    python -W ignore ${NEURALSP_ROOT}/utils/extract_feats.py \
    --data ${data}/${x} \
    --sample_rate ${sample_rate} \
    --feature_type ${feature_type} \
    --feature_config ${feature_conf} \
    --feature_dump_location ${data}/${feature_type}/${x} \
    --jobs ${jobs} \
    --cutset_yaml ${data}/${x}/cutset.yaml \
    --storage_type Lilcom \
    --cmvn Load_Apply \
    --cmvn_location ${data}/${feature_type}/${train_set}/stats.npy
    # lhotse feat extract -j ${nj} \
    #     -r ${CSJ_ROOT} \
    #     ${OUTPUT_PATH}/audio_${part}.json \
    #     ${OUTPUT_PATH}/feats_${part}
    # # Create cuts out of features
    # lhotse cut simple \
    #     -s ${OUTPUT_PATH}/supervisions_${part}.json \
    #     ${OUTPUT_PATH}/feats_${part}/feature_manifest.json.gz \
    #     ${OUTPUT_PATH}/cuts_${part}.json.gz
done

# # Processing complete - the resulting YAML manifests can be loaded in Python to create a PyTorch dataset.
