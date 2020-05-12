
---Package used
tensorflow-cpu==1.15.0
keras==2.4.2
scikit-learn==0.22.1
numpy==1.16.2
nltk==3.4.5
gensim==3.8.0


---File contains
CNN-GS-Selected-text.zip : For "Selected Text" column in dataset, use CNNs with GridSearch and early stopping. Fitting with best parameters combination TO do the sentiment classification


CNN-GS-text.zip :  For "Text" column in dataset,  use CNNs with GridSearch and early stopping. Fitting with best parameters combination STEP BY STEP

GS-Result: Save the GridSearch score for part of hyper paramters combination.


Seq2Seq_Attention.zip: For "Text" column in dataset, use S2S Model and Attention mechanism to summarize "Selected Text" columns; Compute the BELU Score


myutils_V4.py: the utils function in the code; It has to be placed on the same directory of the ipython file.


TwoTasksRNNs.zip: For "Text" and "Selected Text" column in dataset, use combined RNNs Model to finish summarization and classification together.  (The summarization loss in the middle is added to the final loss)

Other graphs: Model structure








