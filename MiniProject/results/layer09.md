PCA of dataset<br>
![Alt text](image.png)<br>
<br>
Label 1 -><br>
    Initial model<br>
        SVC <br>
        hyperparameters -> {kernel="rbf", C=20, gamma="scale"}<br>
        Valid set Accuracy:  0.9626666666666667<br>
        Precision, Recall and F1 Score: (0.9658712615443265, 0.9626666666666667, 0.9626405604263418, None)<br>
    After PCA<br>
    ![Alt text](image-1.png)<br>
        n_components = 70<br>
        SVC<br>
        hyperparameters -> {kernel="rbf", C=20, gamma="scale", degree=5}<br>
        Valid set Accuracy: 0.9346666666666666<br>
        Precision, Recall and F1 Score: (0.9378507574297047, 0.9346666666666666, 0.9346067555105009, None)<br>
<br>
Label 2 -><br>
    After PCA<br>
    ![Alt text](image-2.png)<br>
        n_components = 180<br>
        SVC<br>
        hyperparameters ->  {'kernel': 'rbf', 'gamma': 'scale', 'degree': 5, 'C': 30}<br>
        Valid set Accuracy: 0.9334239130434783<br>
        Precision, Recall and F1 Score: (0.9356685266113715, 0.9334239130434783, 0.9333604351244102, None)<br>
<br>
Label 3 -><br>
    After PCA<br>
    ![Alt text](image-3.png)<br>
        n_components = 32<br>
        SVC<br>
        hyperparameters -> {'C': 10}<br>
        Valid Set Accuracy: 0.9946666666666667<br>
        Precision, Recall and F1 Score: (0.9946607338017173, 0.9946666666666667, 0.9946521329001151, None)<br>
<br>
Label 4 -><br>
    After PCA<br>
    ![Alt text](image-4.png)<br>
        n_components = 130<br>
        SVC<br>
        hyperparameters -> {'C': 10}<br>
        Valid Set Accuracy:  0.9733333333333334<br>
        Precision, Recall and F1 Score: (0.9738508140225787, 0.9733333333333334, 0.9726425543433834, None)<br>