# masinsko_ucenje
Kaggle dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/

# Uvod i cilj projekta

U oblasti mašinskog učenja, posebno u vezi sa computer vision-om, konvolutivne neuronske mreže (CNN) predstavljaju ključnu tehnologiju koja omogućava efikasnu obradu slika i prepoznavanje oblika. 
U ovom projektu testirane su razlicite arhitekture konvolutivnih neuronskih mreza kako bi se napravila klasifikacija MRI slika mozga u cilju prepoznavanja tri vrste tumora mozga kao I prepoznavanje da li je tumor mozga detektovan na slici. Koriscen je skup podataka pod imenom “Brain Tumor Classification (MRI)”.

# Rezultati CNN modela

U nasatvku su prikazani rezultati CNN modela sa razlicitim tehnikama regularizacije, 
Konstruisana su tri modela sa razlicitim tehnikama regularizacije:
    1. Model sa konvolutivnim slojevima
    2. Model sa slojevima agregacije
    3. Dropout model
    
# Konvolutivni slojevi

Kreiran je CNN model od nuie sa konvolutivnim slojevima:
Sa 10 epoha:
Epoch 1/10 - Loss: 1.3511, Accuracy: 0.3624, Validation Loss: 1.1387, Validation Accuracy: 0.5122
Epoch 2/10 - Loss: 1.2081, Accuracy: 0.4499, Validation Loss: 1.0144, Validation Accuracy: 0.6045
Epoch 3/10 - Loss: 1.0997, Accuracy: 0.5327, Validation Loss: 0.9994, Validation Accuracy: 0.5627
Epoch 4/10 - Loss: 1.0086, Accuracy: 0.5636, Validation Loss: 1.1056, Validation Accuracy: 0.4948
Epoch 5/10 - Loss: 0.9399, Accuracy: 0.6002, Validation Loss: 0.8188, Validation Accuracy: 0.6725
Epoch 6/10 - Loss: 0.9198, Accuracy: 0.5936, Validation Loss: 0.7861, Validation Accuracy: 0.6882
Epoch 7/10 - Loss: 0.8433, Accuracy: 0.6468, Validation Loss: 0.7514, Validation Accuracy: 0.6760
Epoch 8/10 - Loss: 0.8329, Accuracy: 0.6485, Validation Loss: 0.9195, Validation Accuracy: 0.5958
Epoch 9/10 - Loss: 0.7973, Accuracy: 0.6551, Validation Loss: 0.6411, Validation Accuracy: 0.7317
Epoch 10/10 - Loss: 0.7477, Accuracy: 0.6777, Validation Loss: 0.7673, Validation Accuracy: 0.6899

Tacnost na test skupu sa Konvolutivnim slojevima: 0.3807106614112854

Sa 50 epoha

Tacnost na test skupu sa Konvolutivnim slojevima: 0.5025380849838257

Sa 100 epoha 
Epoch 100/100
72/72 [==============================] - 89s 1s/step - loss: 0.1981 - accuracy: 0.9229 - val_loss: 0.1944 - val_accuracy: 0.9408
13/13 [==============================] - 2s 170ms/step - loss: 3.7045 - accuracy: 0.6497

Tacnost na test skupu sa Konvolutivnim slojevima: 0.6497461795806885

# Slojevi agregacije

Epoch 1/10 - Loss: 1.3766, Accuracy: 0.3397, Validation Loss: 1.2214, Validation Accuracy: 0.4965
Epoch 2/10 - Loss: 1.2869, Accuracy: 0.4003, Validation Loss: 1.2033, Validation Accuracy: 0.4582
Epoch 3/10 - Loss: 1.2786, Accuracy: 0.4055, Validation Loss: 1.1569, Validation Accuracy: 0.4983
Epoch 4/10 - Loss: 1.2616, Accuracy: 0.4307, Validation Loss: 1.1628, Validation Accuracy: 0.5174
Epoch 5/10 - Loss: 1.2442, Accuracy: 0.4321, Validation Loss: 1.1868, Validation Accuracy: 0.4895
Epoch 6/10 - Loss: 1.2423, Accuracy: 0.4634, Validation Loss: 1.1063, Validation Accuracy: 0.5261
Epoch 7/10 - Loss: 1.2523, Accuracy: 0.4098, Validation Loss: 1.1189, Validation Accuracy: 0.5192
Epoch 8/10 - Loss: 1.2357, Accuracy: 0.4412, Validation Loss: 1.1503, Validation Accuracy: 0.5226
Epoch 9/10 - Loss: 1.2317, Accuracy: 0.4495, Validation Loss: 1.2203, Validation Accuracy: 0.4216
Epoch 10/10 - Loss: 1.2315, Accuracy: 0.4486, Validation Loss: 1.1664, Validation Accuracy: 0.4826

Tacnost na test skupu sa MaxPooling2D  slojevima:  0.20304568111896515

# Dropout

Epoch 1/10 loss: 1.3698 - accuracy: 0.3410 - val_loss: 1.1762 - val_accuracy: 0.4617 
Epoch 2/10 loss: 1.2372 - accuracy: 0.4207 - val_loss: 1.1078 - val_accuracy: 0.5192 
Epoch 3/10 loss: 1.1398 - accuracy: 0.4961 - val_loss: 0.9607 - val_accuracy: 0.5854 
Epoch 4/10 loss: 1.0828 - accuracy: 0.5301 - val_loss: 0.9360 - val_accuracy: 0.6568 
Epoch 5/10 loss: 1.0270 - accuracy: 0.5501 - val_loss: 0.8488 - val_accuracy: 0.6620 
Epoch 6/10 loss: 0.9898 - accuracy: 0.5806 - val_loss: 0.7632 - val_accuracy: 0.6603 
Epoch 7/10 loss: 0.9424 - accuracy: 0.5941 - val_loss: 0.7452 - val_accuracy: 0.6969 
Epoch 8/10 loss: 0.8812 - accuracy: 0.6198 - val_loss: 0.7083 - val_accuracy: 0.6794 
Epoch 9/10 loss: 0.8695 - accuracy: 0.6206 - val_loss: 0.6940 - val_accuracy: 0.7073 
Epoch 10/10 loss: 0.8407 - accuracy: 0.6341 - val_loss: 0.7031 - val_accuracy: 0.7073 

Tacnost na test skupu sa Dropout-om:: 0.32487308979034424
