git clone https://github.com/tesseract-ocr/tesstrain.git

unzip -q scalebar-training-data.zip

mv ./scalebar-training-data/ ./tesstrain/data/scalebar-ground-truth

cd tesstrain

make training MODEL_NAME=scalebar START_MODE=eng TESSDATA=/usr/share/tesseract-ocr/4.00/tessdata

cp ./data/scalebar.traineddata /usr/share/tesseract-ocr/4.00/tessdata/scalebar.traineddata