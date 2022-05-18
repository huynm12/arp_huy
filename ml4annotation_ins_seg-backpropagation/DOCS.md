# End2End Panoptic Segmentation

## Current results
This [Google sheets](https://docs.google.com/spreadsheets/d/1r8cIIylzPhtNIp_q6T3Vcop6SJcH4R8IAKXMIYT8hQo/edit#gid=1210007758) contains experiments details and results.

The information of tabs:
- **Datasets**: information about datasets
- **Experiments**: experiments' description
- **Results**: results of experiments (exp4, exp5 were run with the old exp3. You need to tune parameters for exp3 before runnning the exp4 and exp5 again)

## Datasets

The datasets are symlinked to the datasets folder to be easy to check and train. There are three main folders:

| directory          | root path                                      | datasets |
|--------------------|------------------------------------------------|----------|
| 40k_full           | /home/tungnd13/quynhpt29/v2                    | D, F     |
| 210713_7k_auditted | /home/tungnd13/tuanpa36/audit_20210713_mapping | A, E     |
| 210803_2k_auditted | /home/tungnd13/tuanpa36/to_audit/auditted      | C        |

## Checkpoints

All checkpoints are stored in this [directory](/raid/tungnd13/ml4annotation_models/ml4annotation_end2end_panoptic/outputs) on Hanoi machine.

These are the training configs and weight paths for experiments

| Experiment name | Config                                                      | Weight                              |
|-----------------|-------------------------------------------------------------|-------------------------------------|
| Experiment 1    | configs/vinai_panoptic_210806/exp1_semantic_Mscale_OCR.yaml | vinai_panoptic_exp1/final_state.pth |
| Experiment 2    | configs/vinai_panoptic_210806/exp2_instance_H32.yaml        | vinai_panoptic_exp2/final_state.pth |
| Experiment 3    | configs/vinai_panoptic_210806/exp3_joint.yaml               | vinai_panoptic_exp3/final_state.pth |
| Experiment 3 + CE weight | configs/vinai_panoptic_210806/exp3_joint_instance_ce_weight.yaml | outputs/vinai_panoptic_exp3_instance_ce_weight/final_state.pth |
| Experiment 4    | configs/vinai_panoptic_210806/exp4_semantic_Mscale_OCR.yaml | vinai_panoptic_exp4/final_state.pth |
| Experiment 5    | configs/vinai_panoptic_210806/exp5_instance_H32.yaml        | vinai_panoptic_exp5/final_state.pth |

## Additional configs for training
Besides the default configs of original repository, we have added some additional parameters:

- `MODEL.PANOPTIC_DEEPLAB.INSTANCE.SEMANTIC_EMBED_CHANNELS`: 64, number of embedding channels for foreground semantic input.
- `MODEL.PANOPTIC_DEEPLAB.INSTANCE.USE_SEMANTIC`: True, using semantic as input for instance branch.
- `MODEL.PANOPTIC_DEEPLAB.SEMANTIC.PRETRAINED`: '', pretrained weight for semantic branch.
- `MODEL.PANOPTIC_DEEPLAB.SEMANTIC.IGNORE_HEAD`: True, ignore semantic head when loading pretrained weight.
- `MODEL.PANOPTIC_DEEPLAB.FREEZE_MODULE`: 'instance', or 'semantic', freeze instance branch or semantic branch when training.

## Additional configs for testing
Because the testing procesing takes a lot of time to convert data to COCO format, after each time running the testing, we save preprocessing data to use later. For example, there are two files generated for the test C:
- `datasets/210803_2k_auditted/annotations/instance_val.pkl` for instance evaluation.
- `datasets/210803_2k_auditted/annotations/panoptic_val.json` and a directory `datasets/210803_2k_auditted/annotations/panoptic_val` for panoptic PQ evaluation.

To test the model without running preprocessing, we pass the two file paths that generated from the previous processing to the config files:
- `TEST.INSTANCE_COCO_EVAL`: `datasets/210803_2k_auditted/annotations/instance_val.pkl`
- `TEST.PANOP_JSON`: `datasets/210803_2k_auditted/annotations/panoptic_val.json`

## Reading testing results
We have three different metrics used for evaluate performance of the models:
- IoU for semantic segmentation: can be read on the console.
- AP/AR for instance segmentation: can be read on the console. Moreover, we write down AP to ap_res.txt file. You just need to copy the whole file to the result sheet. However, please remember to clear the file before each evaluation.
- Panoptic Quality (PQ): can be read on the console. This metric is used for tuning parameters

## New processing

We have written the new post processing when we correct some pixels that are misaligned with the semantic labels. You can see that in the function `get_instance_segmentation` in the file [segmentation/model/post_processing/instance_post_processing.py](segmentation/model/post_processing/instance_post_processing.py). To use the old post-processing, just comment the current implementation and uncomment the old ones.

## Loss function (CE Weights)

We have implement the function `get_instance_weight_map` to get the cross-entropy weight of each instance in the file [segmentation/model/post_processing/instance_post_processing.py](segmentation/model/post_processing/instance_post_processing.py). The usage of this function can be found in the `loss` function of the file [segmentation/model/meta_arch/vinai_panoptic_deeplab.py](segmentation/model/meta_arch/vinai_panoptic_deeplab.py).


## Tuning loss weights

One of the problem that we need to do is to tune the balanced weights between losses. We have written a file [tools/tune_params.py](tools/tune_params.py) for tuning the parameters. However, the tuning results are not correct. Maybe you can check the tiny dataset that we used to tune (the test set C) and change with other sets. Note that, you should run the test to get the preprocessed files before tuning.

The sample command for tuning
```
CUDA_VISIBLE_DEVICES=6 RAY_PICKLE_VERBOSE_DEBUG=1 python tools/tune_params.py --cfg configs/vinai_panoptic_210806/tuning/exp3_joint_instance_ce_weight.yaml
```