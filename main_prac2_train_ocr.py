# @brief main_text_ocr
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025
#

import cv2
import os
import pickle
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--train_ocr_path', default="../Materiales_Práctica2/train_ocr", help='Select the training data dir for OCR')
    parser.add_argument(
        '--test_ocr_char_path', default="../Materiales_Práctica2/test_ocr_char", help='Imágenes de test para OCR de caracteres')
    parser.add_argument(
        '--test_ocr_words_path', default="../Materiales_Práctica2/test_ocr_words_plain", help='Imágenes de test para OCR con palabras completas')
    args = parser.parse_args()

    TEST_OCR_CLASSIFIER_IN_CHARS=True
    TEST_OCR_CLASSIFIER_IN_WORDS=True
    SAVED_OCR_CLF = "clasificador.pickle"
    
    # Create the classifier reading the training data
    print("Training OCR classifier ...")

    data_ocr = template_det.data_loaders.OCRTrainingDataLoader()
    if not os.path.exists(SAVED_TEXT_READER_FILE):

        # Load OCR training data (individual char images)
        print("Loading train char OCR data ...")

        # Train the OCR classifier for individual chars
        # clf = .... # POR HACER
        
        with open(SAVED_OCR_CLF, "wb") as pickle_file:
            pickle.dump(clf, pickle_file)

    else:
        with open(SAVED_OCR_CLF, "rb") as pickle_file:
            clf = pickle.load(pickle_file)

    if TEST_OCR_CLASSIFIER_IN_CHARS:
        # Load OCR testing data (individual char images) in args.test_char_ocr_path
        print("Loading test char OCR data ...")
        # gt_test = # POR HACER
        
        print("Executing classifier in char images ...")
        # estimated_test = # POR HACER
        
        # Display of classifier results statistics
        accuracy = sklearn.metrics.accuracy_score(gt_test, estimated_test)
        print("    Accuracy char OCR = ", accuracy)

    if TEST_OCR_CLASSIFIER_IN_WORDS:
        # Load full words images for testing the words reader.
        print("Loading and processing word images OCR data ...")

        # Open results file
        results_save_path = "results_ocr_words_plain"
        try:
            os.mkdir(results_save_path)
        except:
            print('Can not create dir "' + results_save_path + '"')

        results_file = open(os.path.join(results_save_path, "results_text_lines.txt"), "w")
        
        # Execute the OCR over every single image in args.test_words_ocr_path
        # POR HACER ...







