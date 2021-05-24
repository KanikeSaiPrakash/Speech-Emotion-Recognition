# Speech-Emotion-Recognition
## Deep Learning

Follow this README text file to get the clear idea about the repository.

## **TRAINING DATASETS:**
#### _English_
* [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess) - _Toronto Emotional Speech Set_.
* [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) - _Ryerson Audio-Visual Database of Emotional Speech and Song_
* [SAVEE](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee) - _Surrey Audio-Visual Expressed Emotion_
* [CREMA-D](https://www.kaggle.com/ejlok1/cremad) - _Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)_
* [IEMOCAP](https://www.kaggle.com/jamaliasultanajisha/iemocap-full) - _Interactive Emotional Dyadic Motion Capture_

## **TESTING DATASETS:**
#### _German_
* [EMO-DB](https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb) - _Emo-DB Database_


## **DATASET DESCRIPTIONS:**
#### _TESS:_
* There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
* The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format

#### _RAVDESS:_
* This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

###### File naming convention

Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

###### Filename identifiers
- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

#### _SAVEE:_
* The SAVEE database was recorded from four native English male speakers (identified as DC, JE, JK, KL), postgraduate students and researchers at the University of Surrey aged from 27 to 31 years. Emotion has been described psychologically in discrete categories: anger, disgust, fear, happiness, sadness and surprise. A neutral category is also added to provide recordings of 7 emotion categories.
* The text material consisted of 15 TIMIT sentences per emotion: 3 common, 2 emotion-specific and 10 generic sentences that were different for each emotion and phonetically-balanced. The 3 common and 2 × 6 = 12 emotion-specific sentences were recorded as neutral to give 30 neutral sentences. This resulted in a total of 120 utterances per speaker

#### _CREMA:_
* CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

#### _IEMOCAP:_
* The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is an acted, multimodal and multispeaker database, recently collected at SAIL lab at USC. It contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions. It consists of dyadic sessions where actors perform improvisations or scripted scenarios, specifically selected to elicit emotional expressions. IEMOCAP database is annotated by multiple annotators into categorical labels, such as anger, happiness, sadness, neutrality, as well as dimensional labels such as valence, activation and dominance. The detailed motion capture information, the interactive setting to elicit authentic emotions, and the size of the database make this corpus a valuable addition to the existing databases in the community for the study and modeling of multimodal and expressive human communication.
* This files contains the extracted features (text, audio, video, & raw text) of each video clips.

#### _EMO-DB:_
* The EMODB database is the freely available German emotional database. The database is created by the Institute of Communication Science, Technical University, Berlin, Germany. Ten professional speakers (five males and five females) participated in data recording. The database contains a total of 535 utterances. The EMODB database comprises of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral. The data was recorded at a 48-kHz sampling rate and then down-sampled to 16-kHz.

###### Additional Information
Every utterance is named according to the same scheme:

- Positions 1-2: number of speaker
- Positions 3-5: code for text
- Position 6: emotion (sorry, letter stands for german emotion word)
- Position 7: if there are more than two versions these are numbered a, b, c ....

> Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).

Information about the speakers
- 03 - male, 31 years old
- 08 - female, 34 years
- 09 - female, 21 years
- 10 - male, 32 years
- 11 - male, 26 years
- 12 - male, 30 years
- 13 - female, 32 years
- 14 - female, 35 years
- 15 - male, 25 years
- 16 - female, 31 years

## Feature Selection:
- Feature selection aims to obtain a subset of a relevant features that are selected from the feature space and to facilitate subsequent analysis . The traditional feature selection methods are involved extracting features in time or frequency domain. E.g, Pitch, Energy, Power, Amplitude, FFT, Mel-Scale measurements like Mel-spectrum, Mel-frequency Cepstral Coefficients (MFCC), Spectogram, Linear prediction coefficients (LPC), n-gram, Constant Q Cepstral Coefficients (CQCC), etc.
- The most used features of speech involve both time and frequency combined. These features will help in keeping track of both time and frequency variations rather than a limitation to only one domain.
The Simulated and Induced data-set is used in the above approaches rather than the Natural data-set. Natural data involve several pre-processing steps like data cleaning(Noise reduction, sampling issue).etc.  

## PROBLEM DEFINITION:
- There are several traditional statistical algorithms for speech emotion recognition which are limited to an assumption of using the training and testing data from the same database (Cross-corpus problem). Also for feature selection  process CNN makes all low to high-level feature extractions simpler and more efficient but can't deal well in numerous emotional classes and require a lot of data collection to train a model.

## MFCC: 
- They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal cepstrum.
-  The steps to calculate the MFCC are as follows :
   - Take the Fourier transform of (a windowed) a signal. 
   - Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows. 
   - Take the logs of the powers at each of the mel frequencies. 
   - Take the discrete cosine transform of the list of mel log powers. 
   - The MFCCs are the amplitudes of the resulting spectrum.

#### ** _Mel(f) = 2595 log(1+f/700)_ ** 

[Figure MFCC Block diagram](./images/mfcc.png)

## Mel-spectrogram:
- Extracting the Mel-spectrogram features of an audio samples.
   - By mapping the audio signal from time domain to the frequency domain by using fast fourier transform. These operations done frame wise (window filters) by overlapping a portion of adjacent frames.
   -  By converting the y-axis (frequency) to a log scale and the color dimension (amplitude) to decibels to form the spectrogram.
   - Map the y-axis (frequency) onto the mel scale to form the Mel-spectrogram.

[Figure Mel-spectrogram](./images/mel.png)

## Modified GD gram:
- Frame wise group delay after mapping the audio signal to the frequency domain using FFT by overlapping windows and saved in the form of image as shown in figure below.

[Figure Modified GD gram](./images/gd.png)
 

 
## Resnet:

- ResNet is a short name for a residual network. Deep convolutional neural networks have achieved the human level image classification result. Deep networks extract low, middle and high-level features and classifiers in an end-to-end multi-layer fashion, and the number of stacked layers can enrich the “levels” of features. When the deeper network starts to converge, a degradation problem has been exposed. With the network depth increasing, accuracy gets saturated and then degrades rapidly. Such degradation is not caused by overfitting or by adding more layers to a deep network leads to higher training error.
- The deterioration of training accuracy shows that not all systems are easy to optimize.To overcome this problem, Microsoft introduced a deep residual learning framework. Instead of hoping every few stacked layers directly fit a desired underlying mapping, they explicitly let these layers fit a residual mapping. The formulation of F(x)+x can be realized by feedforward neural networks with shortcut connections. Shortcut connections are those skipping one or more layers shown in figure.

- By using the residual network, there are many problems which can be solved such as:
    - ResNets are easy to optimize.
    - ResNets can easily gain accuracy from greatly increased depth.
[Figure Resnet Shortcut connection](./images/resnet.png)

## Work Done:
#### Model 1: 

8 Classes (neutral, calm, happy, sad, anger,fear,Disgust, surprise)

- A CNN Model 1 is built as shown in figure named Model-1, the convolutional layers are 1-Dimensinal as input data shape is aligned 40 features in 1-Dimension and 'ReLu' activation function is used because ReLu function shows the better performance. Dropout of 10 percent is used to get rid of overfitting problem. After that maxpooling layer is used to extract the features that are dominant after the convolutional layers. Finally, flatten the features and feed forward to the dense layer with 'Softmax' activation function to classify.
- 
[Figure CNN Model 1](./images/model1.PNG)
#### Model 2:
- A CNN Model 2 is built as shown in figure named model-2, the convolutional layers are 1-Dimensional and 'ReLu' activation is used. Batch Normalization is used to standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. Dropout of 10 percent is used to overcome the overfitting problem. Then a maxpooling layer is used to extract the dominant features. Finally flatten the features and feed forward to dense layer with 'Softmax 'activation layer to classify.
[Figure CNN Model 2](./images/model3.PNG)
> Model 2 has more hidden units than Model 1, leads to get low training error but still have high generalization error due to 'overfitting' and 'high variance' result in accuracy reduction.

- The size of Mel-spectrogram feature is 224 x 224 x 3 image, which is used to train the few standard pretrained models such as Resnet-50, Resnet-34 and Resnet-18. These models are intially loaded with  weights set to the Imagenet standard called Transfer Learning, by removing the top layer i.e, softmax activation layer where standard models are used for 1000 classes and we are adjusting it to our convenience by removing the top layer and adding a trainable dense(softmax activation) layers to classify less number of classes. In this project  Resnet-50 is trained on Mel-spectrogram features for 4(neutral, happy, sad, anger), 5(neutral, happy, sad, anger,fear), 6(neutral, happy, sad, anger,fear,Disgust) emotional classes. It is observed that as the number of classes increases the performace of the model decrases w.r.to validation accuracy as well as the testing accuracy on testing with Emo-db dataset. In the same way the Resnet-34 and Resnet-18 are trained with mel-spectrogram features for 6 classes and it is observed that the Resnet-34 is performing better than Resnet-18 w.r.to validation accuracy. And testing of Emo-db dataset has been done with only Resnet-34 as it is performing better and rest of the works done only Resnet50.


- Further work involves the combination of the two features so that it can add more weight to the features. Another feature is Modified Gd-gram as shown in figure below  and it is extracted for all the audio samples and trained the Resnet-50 for 6 classes by combining both features framewise and resize it to size of 224 x 224x 3(Standard Imagenet input size). The performance of this model is less than model trained with Mel-spectrograms alone. 

- This is observed that the performance reduced due to the image concatenated with other image and resize factor to make it standard size. To get rid of this effect, using a seperate model to each type of feature image i.e Mel-spectrogram features to Resnet-50 pretrained model and Modified-Gd-gram to VGG-16 pretraied model and both models are concatenated as shown in figure below at the last layers called Multi-input method. The inputs to the Multi-input model can be of any type (e.g: image, categorical, numerical vector) etc. 
[Figure Mel_and Modified Gd Gram Multi input model](./images/melGD-multi.PNG)

- In similar way the multi-input model can be applicable to any type of input data, so this model will make use of Mel-spectrogram to Resnet-50 and MFCC's to the CNN model and both concatenated as shown in figure below at the last layer is performing better than the mutli-input model of mel-spectrogram and Modified-Gd-gram, where as it's performance is not upto the mark of the model with Mel-spectrograms alone.
[Figure Mel_and MFCC Multi input model](./images/melmfcc.PNG)
## Results:

- Combined Ravdess and Tess datasets. Models are trained with 80 percent data and tested with remaining. The CNN Model 1 trained with Batch size of 16 and 200 Epochs, result in accuracy on an average of '85.82 percent' as shown in accuracy and loss figures (model accuracy). The CNN Model 2 trained with the same Batch size and Epochs Epochs, result in accuracy on an average of '82.05 percent' as shown in figures below (model accuracy).

[Figure CNN model 1 accuracy](./images/model1_acc.png)
[Figure CNN model 1 loss](./images/model1_loss.png)
[Figure CNN model 2 accuracy](./images/model_2_acc.png)
[Figure CNN model 2 loss](./images/model2_loss.png)

- Results have been obtained for emodb test dataset for different models as summarised in Table 1. Details of the features and number of classes are also given in the Table.

## TABLE:

| **Models**     |**Classes**        |**Features**       |**Datasets(English)**       |**Validation Accuracy**           |**Testing Accuracy on Emodb German dataset**|
| ---------      |   -------         | -------   ----    |  ------------------------  | ----------------------------     |  ----------------------                    |
|   CNN Model 1      |    8    |     MFCC    |       Ravdess, Tess       |    85.82        %  |         % |
|   |    |   |    |    |    |
|   CNN Model 1      |    8    |    MFCC     |        Ravdess, Tess        |     82.05        %  |           % |
|   |    |   |    |    |    |
|   Resnet 50      |   4     |    Mel-spectrogram     |      Ravdess, Tess, Crema, Iemocap, Savee        |    75.76        %  |      64.60     % |
|   |    |   |    |    |    |
|    Resnet 50       |    5    |     Mel-spectrogram    |      Ravdess, Tess, Crema, Iemocap, Savee         |    66.86        %  |      46.57     % |
|   |    |   |    |    |    |
|    Resnet 50       |    6    |    Mel-spectrogram     |Ravdess, Tess, Crema, Iemocap, Savee |     62.18       %  |     35.46      % |
|   |    |   |    |    |    |
|   Resnet 34        |    6    |    Mel-spectrogram     |         Ravdess, Tess, Crema, Iemocap, Savee      |    68.15        %  |    48.33  % |
|   |    |   |    |    |    |
|   Resnet 18       |     6   |     Mel-spectrogram   |       Ravdess, Tess, Crema, Iemocap, Savee        |     67.62       %  |           % |
|   |    |   |    |    |    |
|    Resnet 50       |    6    |    Mel-spectrogram & Modified Gd gram framewise concatention |      Ravdess, Tess, Crema, Iemocap, Savee         |    65.67   %  |   32.82        % |
|   |    |   |    |    |    |
|   Resnet 50 and VGG 16       |     6   |   Mel-spectrogram & Modified GD gram     |        Ravdess, Tess, Crema, Iemocap, Savee       |   65.01         %  |  31.72         % |
|   |    |   |    |    |    |
|    Resnet 50 and CNN Model 1      |    6    |    Mel-spectrogram &MFCC    |       Ravdess, Tess, Crema, Iemocap, Savee        |    73.14        %  |    39.65       % |
|   |    |   |    |    |    |


## Conclusion :
- We explored three features such as Mel-spectrogram, Modified-Gd gram, MFCC's. In which Mel-spectrogram and MFCC's are considered as the primary features to train the model. With the results obtained Resnet-50 performs better for classes where there is no confusion e,g: For 4(neutral, happy, sad, anger) it performs well as all emotions differ in the feature image pattern where as by involving fear and disgust emotions the model is getting confuse sue to similarities in pattern of anger, fear, disgust.

- Mel-spectrogram gives the better result on features category and Resnet-50 and Resnet-34 gives the better results in 4 and 6 classes respectively. 

- In overall Resnet-34 (34-layers) performs well compared to Resnet-50 (50-layers) for high number of classes. As the number of cnn layers increases the high level features(large area of image) are considered as the network goes deeper, this results in confusion due to the similarities in patterns of feature images of certain classes e.g: 1) Anger, Fear, Disgust. 2) Neutral, Sad.


- The performance of Multi-input model considering MFCC's and Mel-spectrogram is better than the model with Mel-spectrogram alone and multi-input model with Modified-Gd gram and Mel-spectrogram as Mfcc's are features that are coefficients of nearest or respective windowed mel-scale coefficient. So MFCC's and Mel-spectrogram are the best features to evaluate emotions in speech.
