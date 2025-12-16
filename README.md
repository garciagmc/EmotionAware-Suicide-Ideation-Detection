# EmotionAware-Suicide-Ideation-Detection
**Suicidal Ideation Detection via Temporal Emotional Dynamics Analysis in Text Notes**
| Detail | Value |
| :--- | :--- |
| **Author(s)** | García-Galindo, María del Carmen; Hernández-Castañeda, Ángel |
| **Affiliation** | Autonomous University of Mexico State (UAEMex) |
| **Date** | December 2025 |

## Description

This study presents a Machine Learning (ML) approach for Suicidal Ideation (SI) detection by analyzing the temporal emotional dynamics present in text notes. Unlike static methods, our model considers the emotional shifts (dynamics) between notes as a key factor for risk classification. The proposed pipeline consists of five main modules: preprocessing, feature extraction, latent emotion identification, temporal analysis of emotional dynamics, and final classification.

## Ethical Considerations

**IMPORTANT:** This project is solely for academic research purposes.

* **No Substitute for Professional Diagnosis**: The results of this model should NOT be used for clinical decision-making or diagnosis.
* **Data Privacy**: Sensitive data used for training is not included in this repository. Only the code for processing and modeling is provided.
* **Responsible Use**: Users are required to utilize this code ethically and responsibly, adhering to all relevant privacy and mental health regulations.

## System Requirements and Dependencies

To replicate the results of this research, you need **Python 3.6** and the following libraries.

### 1. Environment Setup
  
1. **Clone the repository:*
   ```bash
   git clone https://github.com/garciagmc/EmotionAware-Suicide-Ideation-Detection.git
   cd EmotionAware-Suicide-Ideation-Detection ```

2. **Create and activate the virtual environment (Conda recommended):**
   ```bash
   conda create --name emotionAware python=3.6
   conda activate emotionAware ```
   
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt ```
  
## Required Data Structure

The code expects the input corpora (not included) to be provided as CSV files with a specific format.

### Input File Format
* `Corpora/suicide/consummated_suicide.csv`
* `Corpora/suicide/suicidal_intention.csv`
* `Corpora/suicide/not_Suicide.csv`
* `Corpora/emotions/CEASE.csv`

### Data Format

* **Delimiter:** The input files must use the **Hash (`#`)** symbol as the column delimiter.
* **Required Columns:** The files must contain the following columns: `Texts` and `Label`.
 

# Workflow: Training and Replication

This guide outlines the six sequential steps required to replicate the full pipeline, from corpus preprocessing to final classification.

### Step 1: Data Preparation and Splitting

The raw corpora are processed to separate the text content from their respective labels into distinct `.txt` files.

* **Action:** Run the appropriate data adaptation script.
* **Scripts:** `processCEASE.py` or `processSuicide.py`
* **Location:** Execute from the `adaptData/` folder.
* **Output:** Generates separate files for raw texts (sentences) and corresponding labels in the `emotionsCEASE/` or `suicide/` target folders.

### Step 2: Generate Embeddings for the Emotion Classifier

Feature vectors (embeddings) are generated for the CEASE corpus to train the intermediate emotion classifier.

* **Goal:** Create feature vectors from the CEASE sentences.
* **Location:** Execute the script from the `Embeddings/<type of embedding-TFIDF/Doc2Vec/LDA-LSI>` directory.
* **Example Command (TF-IDF):**
    ```bash
    python3 generateTfidfVectors.py ../../adaptData/emotionsCEASE/sentencesCEASE.txt ../../adaptData/emotionsCEASE/labelsCEASE.txt ../../adaptData/emotionsCEASE/
    ```
* **Output:** Two binary files (`.obj` extension) containing the vectorized representations of the texts.

### Step 3: Train the Emotion Classifier

The vectors generated in Step 2 are used to train the emotion classifier.

* **Goal:** Train a model to classify the 15 defined emotions.
* **Location:** Execute the script from the `suicideProject/sentimentAnalisys/` directory.
* **Command:**
    ```bash
    python3 trainSentimentClassifier.py
    ```
* **Encoded Emotion Order (Output Indexing):**
    ```
    ['Abuse\n' 'Anger\n' 'Blame\n' 'Fear\n' 'Forgiveness\n' 'Guilt\n' 'Peacefulness\n' 'Hopefulness\n' 'Hopelessness\n' 'Information\n' 'Instruction' 'Love\n' 'Pride\n' 'Sorrow\n' 'Thankfulness\n']
    ```
* **Output:** A table showing the F-measure obtained and the trained model saved for later use.

### Step 4: Generate Sentence-Level Vectors for Suicide Data

Feature vectors for the suicide and non-suicide ideation notes are generated at the **sentence level** for the LSTM input sequence.

* **Goal:** Create sentence-level vectors for the suicide and not_suicide texts.
* **Location:** Execute the script from the `Embeddings/<type of embedding-TFIDF/Doc2Vec/LDA-LSI>` directory.
* **Commands (Separate Execution):**
    ```bash
    # For suicide texts
    python3 loadTfidfVectorsSentLevel.py ../../adaptData/suicide/sentencesSuicideIdeation.txt ../../adaptData/suicide/labelsSuicideIdeation.txt ../../adaptData/suicide/Suicide

    # For Not_suicide texts
    python3 loadTfidfVectorsSentLevel.py ../../adaptData/suicide/sentencesNotSuicideIdeation.txt ../../adaptData/suicide/labelsNotSuicideIdeation.txt ../../adaptData/suicide/NotSuicide
    ```

### Step 5: Load Sentiment Model and Generate Emotion Vectors

The sentiment model trained in Step 3 is loaded to process the suicide data vectors (from Step 4) and generate **emotion vectors** for each text, representing the dynamic sequence.

* **Goal:** Transform sentence vectors into emotion vectors using the pre-trained sentiment model.
* **Location:** Execute the script from the `suicideProject/sentimentAnalisys/` directory.
* **Script:** `loadSentimentModel.py`
* **Commands:**
    ```bash
    # For not_suicide
    python3 loadSentimentModel.py NotSuicideIdeationSentiments ../adaptData/suicide/NotSuicideembedsSent.obj

    # For suicide_texts
    python3 loadSentimentModel.py suicideIdeationSentiments ../adaptData/suicide/SuicideembedsSent.obj
    ```

### Step 6: Final Suicide Ideation Classification

The emotion vectors (the dynamic features) are used as input to the main LSTM classification model.

* **Goal:** Train and evaluate the Suicide Ideation Detector (LSTM). 
* **Location:** Execute the script from the `suicideProject/lstm/` directory.
* **Command:**
    ```bash
    python3 suicideDetector.py
    ```
