# ✊✋✌️ Riconoscimento Gesti della Mano: Sasso, Carta, Forbice

Progetto finale del Modulo 4 (Epicode): una **Rete Neurale Convoluzionale (CNN)** con **Data Augmentation**, addestrata sul dataset *Rock-Paper-Scissors (CGI)*, che classifica le immagini della webcam in **Sasso**, **Carta** o **Forbice**.

## Contenuto

| File | Descrizione |
|---|---|
| `rock_paper_scissors_cnn.ipynb` | Notebook completo (consigliato Google Colab con GPU) |

## Struttura del notebook

1. **Download del dataset** con `kagglehub` (`sanikamal/rock-paper-scissors-dataset`), con fallback automatico su `rps.zip` / `rps-test-set.zip`. Le cartelle `rock/`, `paper/`, `scissors/` vengono trovate in automatico.
2. **Fase A: Data pipeline**: `ImageDataGenerator` con `rescale=1./255`, `rotation_range=40`, `width_shift_range=0.2`, `shear_range=0.2`, `horizontal_flip=True`, `fill_mode='nearest'`; caricamento a batch con `flow_from_directory`. Split 80/20 del training set per la validazione; il test set è usato solo alla fine.
3. **Fase B: CNN sequenziale**: input 150×150×3 → 4 blocchi `Conv2D`+`MaxPooling2D` (32→64→128→128) → `Flatten` → `Dropout(0.5)` → `Dense(512, relu)` → `Dense(3, softmax)`.
4. **Fase C: Training**: `RMSprop`, `categorical_crossentropy`, `accuracy`, 20 epoche; callback personalizzata che si ferma oltre il 98% di accuratezza, più `EarlyStopping` e `ModelCheckpoint`.
5. **Fase D: Test "wild"**: valutazione sul test set (classification report e matrice di confusione), scatto dalla webcam in Colab o caricamento di foto, confronto quantitativo CGI vs foto reali e **analisi del domain gap**.

## Come eseguire

1. Apri `rock_paper_scissors_cnn.ipynb` in [Google Colab](https://colab.research.google.com/).
2. `Runtime → Cambia tipo di runtime → GPU`.
3. `Runtime → Esegui tutto`. Alla cella della webcam, consenti l'accesso alla fotocamera e scatta la foto della tua mano davanti a uno **sfondo chiaro e uniforme**.

Requisiti in locale: `tensorflow`, `kagglehub`, `matplotlib`, `scikit-learn`, `pillow`.
