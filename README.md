# ImageDataExtractor 2.0

## Segmentation

### TODO

- Add segmentation module.
- fix paths for loading pytorch model in segmentation.

## Analysis

### TODO

- add analysis module:
    - particle size histogram
    - radial distribution functions

## OCR

### TODO

- ~~add greek language dictionary/model. Consider how this should be done automatically when someone pip installs/installs from source.~~ No need. Fine tuning english language model should suffice, labelling μ's as u's.
- ~~extract scalebar images using the text detector.~~
- ~~finetune ocr model.~~
- fix paths in text detection.

### Useful resources

- [Tesstrain](https://github.com/tesseract-ocr/tesstrain) is all that is needed to very simply finetune a Tesseract OCR model. 

- [Tesseract documentation](https://github.com/tesseract-ocr/tessdoc)

- [Finetuning Tesseract](https://github.com/tesseract-ocr/tessdoc/blob/master/TrainingTesseract-4.00---Finetune.md) 

- [Finetuning Tesseract simpler](https://medium.com/@guiem/how-to-train-tesseract-4-ebe5881ff3b7) 

- ["Optimal image resolution (dpi/ppi) for Tesseract 4.0.0 and eng.traineddata?"](https://groups.google.com/forum/#!msg/tesseract-ocr/Wdh_JJwnw94/24JHDYQbBQAJ) This is a useful resource for optimising the performance of OCR (doesn't work with Cam google account, view in incognito).