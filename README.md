# EEG-fNIRs_CLIP
This is the source code containing the code to process and evaluate an EEG+fNIRs CLIP model from an auditory task involving a simple but large 1-second tone presentation. This Contrastive Learning  (CL) evaluation suggests a well-defined subject-level clusters, using both modalities and generating projections using UMAP and t-SNE.

A peformance evaluated in the [Steimetzger et al 2022](https://www.sciencedirect.com/science/article/pii/S0378595520303403?via%3Dihub) dataset shows a considerable level of performance for HbO pos/neg condition classification based on the following barplots:

![results_barplots_best](https://github.com/user-attachments/assets/56bbf697-566c-44d6-8f82-df0ec9db8329)

![results_barplots_matrix_Specificity](https://github.com/user-attachments/assets/b99b65b8-b2bd-461a-b8fc-8f5097617267)

![results_barplots_matrix_Sensitivity](https://github.com/user-attachments/assets/5ce42511-7324-4499-9dfa-7b1c78ea18a8)

A cluster representation using the InfoNCE loss (loss) is this. Defining high slihoutte and normalized mutual-information (MI) scores having both modalities

<img src="https://github.com/meiyor/EEG-fNIRs_CLIP/blob/main/gifs/new_Record-ezgif.com-video-to-gif-converter.gif"
     alt="new_Record-ezgif com-video-to-gif-converter"
     width="1200"
     height="1000"/>
     
And after re-writing the lose in a semi-supervised way leaving subject_k features out of the training we obtain the following, with a better separability between conditions.

<img src="https://github.com/meiyor/EEG-fNIRs_CLIP/blob/main/gifs/new_Record-ezgif.com-video-to-gif-converter.gif"
     alt="new_Record-ezgif com-video-to-gif-converter"
     width="1200"
     height="1000"/>
