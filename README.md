# ImageDataExtractor 2.0

## OCR

### TODO

- ~~add greek language dictionary/model. Consider how this should be done automatically when someone pip installs/installs from source.~~ No need. Fine tuning english language model should suffice, labelling μ's as u's.
- extract scalebar images using the text detector.
- finetune ocr model.

### Useful resources

- [Tesseract documentation](https://github.com/tesseract-ocr/tessdoc)

- [Finetuning Tesseract](https://github.com/tesseract-ocr/tessdoc/blob/master/TrainingTesseract-4.00---Finetune.md) 

- ["Optimal image resolution (dpi/ppi) for Tesseract 4.0.0 and eng.traineddata?"](https://groups.google.com/forum/#!msg/tesseract-ocr/Wdh_JJwnw94/24JHDYQbBQAJ) This is a useful resource for optimising the performance of OCR (doesn't work with Cam google account, view in incognito).