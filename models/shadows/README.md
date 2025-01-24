# Shadow correction of aerial imagery
This repository contains the code to perform shadow correction of aerial images in RGB format. It is composed of two parts:
1. **ShadowScout**: shadow detection in RGB or RGB+NIR images
2. **ShadowCorrect**: correct shadows in RGB images based on their shadow masks

The code has been tested on python3.8.

< ![Example image](https://raw.githubusercontent.com/elucidatalab/starterkits/shadows/main/media/example.png) >
![Example image](./main/media/example.png)

## Package installation
Install the *shadows* code package and its dependencies with 

```bash
pip install -e .
```

## 1. ShadowScout

### Test a pretrained model on plain RGB images of the AISD dataset

A pretrained model is stored in the output_AISD directory. To access the full AISD dataset visit:
https://github.com/RSrscoder/AISD

```bash
python src/shadows/shadowscout/model/run_model.py infer --test_dir data/AISD/test/shadow/ --save_dir output_AISD/shadow_masks --config_path output_AISD/shadow_masks/config.json --model_path output_AISD/best_epoch_checkpoint.pt
```
Predicted shadow masks will be saved to output_AISD/predicted_masks/
Ground truth data is located in data/AISD/test/ and the quality of the prediction can be 
evaluated with the functions in shadows.shadowscout.image_manipulation.compute_metrics


### Test a pretrained model on plain 4-band images (RGB + NIR)
```bash
python src/shadows/shadowscout/model/run_model.py infer --test_dir data/4bands/test/ --save_dir output_4bands/shadow_masks --model_path output_4bands/shadow_masks/best_epoch_checkpoint.pt  --config_path output_4band/shadow_masks/config_4band.json
```


### Train the model from scratch
First edit the config file located in src/shadows/shadowscout/config.json. The main parameters to 
edit concern the number of channels (and the initial channel_weights for each channel). The config 
file will be copied to the output directory

**Model training**:
```bash
python src/shadows/shadowscout/model/run_model.py train --train_dir PATH_TO_TRAIN_IMAGES_FOLDER --validate_dir PATH_TO_VALIDATE_IMAGES_FOLDER --save_dir PATH_TO_OUTPUT_DIRECTORY --config_path PATH_TO_CONFIG_FILE
```

For convenience, two config files are provided: one for regular 3-channel inputs (src/shadows/shadowcast/config.json) and one for 4-channel inputs (src/shadows/shadowcast/config_4band.json).

Subsequently, **inference** can be made on a sample of images:
```bash
python src/shadowscout/model/run_model.py infer --test_dir PATH_TO_TEST_IMAGES_FOLDER --save_dir PATH_TO_OUTPUT_DIRECTORY --model_path PATH_TO_MODEL --config_path PATH_TO_CONFIG_FILE
```

Note that being an optimization problem, shadow masks can perfectly be predicted for images in the train and validation datasets as well. Conversely, test images can also be included in the model training.

## 2. ShadowCorrect
This method is based on the paper [Shadow removal method for high-resolution aerial remote sensing images based on region group matching](https://www.sciencedirect.com/science/article/abs/pii/S0957417424016063)

In brief, it compensates the effects of shadows on certain pixels of an image by leveraging the information on their surrounding non-shadow pixels.

Although the method requires several parameters for optimal results, we have enhanced it to automatize the process. As such, the shadow correction model can be run using one of three parameter configuration strategies:
1. *default*, which uses the default settings
2. *manual*, which requires providing a configuration json file with custom parameters
3. *auto*, where parameters are precomputed for groups of similar images. With this approach, a batch of images is clustered based on how similar they are and the shadow correction parameters are optimized for each cluster. These are subsequently stored locally. When correcting a new image, it is first matched to its closest cluster and that cluster's parameter configuration is used for the correction. This is the default behavior of the model.

### Defining parameter configurations for image clusters
In order to (re-)run the  optimization of shadow correction parameters for groups of similar images, run the following code:

```bash
python src/shadows/shadowcorrect/cluster_images.py --image_dir PATH_TO_TRAIN_IMAGES --shadow_mask_dir PATH_TO_COMPUTED_SHADOW_MASKS --output_dir output_4bands/shadow_corrected/
```

This will use the default shadow correction configuration to initiate the process. Optionally, you can also pass a custom configuration json via the *--config_path* parameter. 
The process will output a json file, stored in the location indicated by the *--output_dir* parameter and containing the statistics required to match a new image to its most similar cluster and use that cluster's configuration. Note that for optimal results, the clustering-based parameter optimization should be run on a batch of images with similar characteristics as the images which will be subsequently corrected. Note as well that this process is lengthy.

### Shadow correction
Finally, shadows can be corrected with the following command. Note that the method requires the shadow mask to have been already computed

```bash
python src/shadows/shadowcorrect/run_model.py --image_dir data/4bands/test/ --shadow_mask_dir output_4bands/shadow_masks/predicted_masks/ --output_dir output_4bands/shadow_corrected/ --clustering_parameter_optimization_fpath src/shadows/shadowcorrect/optimized_cluster_parameters.json
```
