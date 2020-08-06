### Instructions for finetuning Tesseract on scalebar data

1. git clone https://github.com/tesseract-ocr/tesstrain into the directory /imagedataextractor/imagedataextractor/scalebar/finetune/

2. unzip scalebar-ground-truth.zip, and place the folder of images into /tessdata/data/ .

3. make training MODEL_NAME=scalebar START_MODEL=eng TESSDATA=/usr/share/tesseract-ocr/4.00/tessdata CORES=8 PSM=6 MAX_ITERATIONS=20000