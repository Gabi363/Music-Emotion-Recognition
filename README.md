# Analysis of the possibilities of processing music in artificial intelligence models with consideration of emotion recognition

## Engineering thesis for AGH Computer Science and Intelligent Systems studies

### Idea
This project investigates various approaches to processing musical signals in the context of automatic emotion 
recognition using machine learning and deep learning methods. The work compares models trained on hand-crafted acoustic 
features with models trained directly on time–frequency representations (mel-spectrograms).

### Content
- acoustic_features_experiments.ipynb - Classification on dataset with Turkish music recordings based on acoustic features.
- spectrograms_experiments.ipynb - Classification on dataset with Turkish music recordings based on mel-spectrograms.
- music_tools.py - Essential functions for audio processing, data augmentation and model evaluation.
- MIR.ipynb - Demonstration and explanation of the most relevant acoustic features used in Music Information Retrieval (MIR).

### Data
The project uses a dataset of Turkish music labeled with four emotional categories.
<br>
Dataset download link: https://doi.org/10.24432/C5JG93 [1].
<br>
Conducted experiments were inspired by the article _Music Emotion Recognition with Machine Learning Based on Audio Features_ [2].

Citation: <br>
[1] Er, M. (2019). Turkish Music Emotion [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JG93
<br>
[2] Er, M. B., & Esin, E. M. (2021). Music Emotion Recognition with Machine Learning Based on Audio Features. Computer Science, 6(3), 133-144. https://doi.org/10.53070/bbd.945894

### How to run project
1. Create a Python environment and install required packages using:
````
pip install -r requirements.txt
````
2. Download the dataset and update the dataset paths at the beginning of the Jupyter notebooks.
3. Run the following notebooks to reproduce or extend the experiments:
   - acoustic_features_experiments.ipynb
   - spectrograms_experiments.ipynb