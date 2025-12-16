# EmotionAware-Suicide-Ideation-Detection
Suicidal Ideation Detection via Temporal Emotional Dynamics Analysis in Text Notes
  Author(s): Garcia-Galindo María del Carmen; Hernández-Catsañeda Ángel 
  Affiliation: Autonomous University of Mexico State                                           Date: December 2025

# Description
This study presents a Machine Learning (ML) approach for Suicidal Ideation (SI) detection by analyzing the temporal emotional dynamics present in text notes. Unlike static methods, our model considers the emotional shifts (dynamics) between notes as a key factor for risk classification. The proposed pipeline consists of five main modules: preprocessing, feature extraction, latent emotion identification, temporal analysis of emotional dynamics, and final classification.

# Ethical Considerations
IMPORTANT: This project is solely for academic research purposes.
No Substitute for Professional Diagnosis: The results of this model should NOT be used for clinical decision-making or diagnosis.
- Data Privacy: Sensitive data used for training is not included in this repository. Only the code for processing and modeling is provided.
- Responsible Use: Users are required to utilize this code ethically and responsibly, adhering to all relevant privacy and mental health regulations.

# System Requirements and Dependencies
To replicate the results of this research, you need Python 3.6 and the following libraries.

- 1 Environment Setup
  git clone https://github.com/garciagmc/EmotionAware-Suicide-Ideation-Detection.git
  cd EmotionAware-Suicide-Ideation-Detection

- 2 Create and activate the virtual environment (Conda recommended):
  conda create --name emotionAware python=3.6
  conda activate emotionAware

- 3 Install dependencies:
  pip install -r requirements.txt

# Required Data Structure
Input File Format
    - Corpora/suicide/consummated_suicide.csv
    - Corpora/suicide/suicidal_intention.csv
    - Corpora/suicide/not_Suicide.csv
    - Corpora/emotions/CEASE.csv
  Delimiter: Hash (#)
  Required Columns (with # separator emphasis): Texts and Label#

# Workflow: This guide outlines the six steps required to process the corpus, train the intermediate emotion classifier, generate temporal emotion features, and finally train the main suicide ideation detector.

- Step 1: Data Preparation and Splitting: The first step is to process the raw corpora (CEASE for emotion, and the suicide ideation corpus) to separate the text content from their respective labels into distinct files.
  Action: Run the appropriate data adaptation script.
  Scripts: processCEASE.py or processSuicide.py
  Location: The scripts are located in the adaptData/ folder.
  Output: The scripts split the CSV data into two file types within their respective target folders (emotionsCEASE or suicide): One file containing all the raw texts (sentences), other file containing all the corresponding labels.

- Step 2: Generate Embeddings for the Emotion Classifier: Embeddings are generated for the CEASE corpus texts to train the intermediate emotion classifier.
  Goal: Create feature vectors from the CEASE sentences.
  Location: Execute the script from the Embeddings/<type of embedding-TFIDF/Doc2Vec/LDA-LSI> directory.
  Example Command (TF-IDF):
        python3 generateTfidfVectors.py ../../adaptData/emotionsCEASE/sentencesCEASE.txt ../../adaptData/emotionsCEASE/labelsCEASE.txt              ../../adaptData/emotionsCEASE/
  Output: Two binary files (.obj extension) containing the vectorized representations of the texts as Python lists.

- Step 3: Train the Emotion Classifier: Using the vectors generated in Step 2, the emotion classifier is trained.
  Goal: Train a model to classify the 13 defined emotions.
  Location: Execute the script from the suicideProject/sentimentAnalisys/ directory.
  Command:
    python3 trainSentimentClassifier.py

  Encoded Emotion Order (Output Indexing): ['Abuse\n' 'Anger\n' 'Blame\n' 'Fear\n' 'Forgiveness\n' 'Guilt\n' 'Peacefulness\n' 'Hopefulness\n' 'Hopelessness\n' 'Information\n' 'Instruction' 'Love\n' 'Pride\n' 'Sorrow\n' 'Thankfulness\n']

  Output: The script returns a table showing the F-measure obtained during training and saves the trained model for later use.

- Step 4: Generate Sentence-Level Vectors for Suicide data: The feature vectors for the suicide and non-suicide ideation notes are now generated. This must be done at the sentence level, as these vectors will serve as the sequence input for the LSTM network.

  Goal: Create sentence-level vectors for the suicide and not_suice texts.
  Location: Execute the script from the Embeddings/<type of embedding-TFIDF/Doc2Vec/LDA-LSI> directory.
  Commands (Separate Execution):
    # For suicide texts: python3 loadTfidfVectorsSentLevel.py ../../adaptData/suicide/sentencesSuicideIdeation.txt           ../../adaptData/suicide/labelsSuicideIdeation.txt ../../adaptData/suicide/Suicide
  # For Not_suicide texts: python3 loadTfidfVectorsSentLevel.py ../../adaptData/suicide/sentencesNotSuicideIdeation.txt ../../adaptData/suicide/labelsNotSuicideIdeation.txt ../../adaptData/suicide/NotSuicide

- Step 5: Load sentiment Model and Generate Emotion Vectors: The sentiment model trained in Step 3 is now loaded to process the suicide data vectors (from Step 4) and generate emotion vectors for each text. These vectors represent the dynamic sequence of emotional changes.

  Goal: Transform the sentence vectors into emotion vectors using the pre-trained sentiment model.
  Location: Execute the script from the suicideProject/sentimentAnalisys/ directory.
  Script: loadSentimentModel.py
  Commands:
    # For not_suicide: python3 loadSentimentModel.py NotSuicideIdeationSentiments ../adaptData/suicide/NotSuicideembedsSent.obj
    # For suicide_texts: python3 loadSentimentModel.py suicideIdeationSentiments ../adaptData/suicide/SuicideembedsSent.obj

Step 6: Final Suicide Ideation Classification: The final step uses the emotion vectors (the dynamic features) as input to the main classification model.
    Goal: Train and evaluate the Suicide Ideation Detector (LSTM).
    Location: Execute the script from the suicideProject/lstm/ directory.
    Command:
    python3 suicideDetector.py

    
  

  
