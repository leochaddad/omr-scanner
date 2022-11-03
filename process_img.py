from encodings.utf_8 import decode
import string
import cv2
import numpy as np
from imutils import contours as imutils_contours
from imutils.perspective import four_point_transform
import base64


def process_img(image):


    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.bilateralFilter(img, 9, 75, 75)
    img_canny = cv2.Canny(img_blur, 150, 50)
    img_thresh = cv2.adaptiveThreshold(img_canny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of contours found = " + str(len(contours)))


    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        area = cv2.contourArea(c)

        if len(approx) == 4 and ar >= 1.0 and ar <= 1.32:
            print("Found the contour")
            break

    
    document = four_point_transform(img, approx.reshape(4, 2))


    h, w = document.shape
    document = document[int(0.02*h):int(0.98*h), int(0.08*w):int(0.92*w)]


    document_blur = cv2.GaussianBlur(document, (7, 7), 0)


    doc_thresh = cv2.threshold(document_blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU, 
        )[1]


    kernel = np.ones((7,7),np.uint8)
    doc_thresh = cv2.dilate(doc_thresh,kernel,iterations = 2)

    contours = cv2.findContours(doc_thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[0]



    question_cnts = []

    MIN_AREA = 4500
    MIN_LEN = 60



    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        area = cv2.contourArea(c)

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        if w >= MIN_LEN and h >= MIN_LEN and ar >= 0.8 and ar <= 1.2 and len(approx) > 8 and area > MIN_AREA:
            question_cnts.append(c)


    numbered = cv2.cvtColor(document.copy(), cv2.COLOR_GRAY2BGR)


    question_cnts = imutils_contours.sort_contours(question_cnts,
        method="top-to-bottom")[0]



    grid = []
    row = []
    for (i, c) in enumerate(question_cnts, 1):
        row.append(c)
        if i % 10 == 0:  
            (cnts, _) = imutils_contours.sort_contours(row, method="left-to-right")
            grid.append(cnts)
            row = []



    i = 0
    for row in grid:
        for c in row:
            cv2.drawContours(numbered, [c], -1, (0, 255, 0), 3)
            cv2.putText(numbered, str(i), (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            i += 1



    question_grid = []
    numbered = cv2.cvtColor(document.copy(), cv2.COLOR_GRAY2BGR)



    for (row) in grid:
        question_grid.append(row[:5])
    for (row) in grid:
        question_grid.append(row[5:])

    for (question_number, row ) in enumerate(question_grid):
        question_number += 1
        for (option_number, c ) in enumerate(row):
            option_letter = chr(ord('A') + option_number)
            cv2.drawContours(numbered, [c], -1, (0, 255, 0), 3)
            cv2.putText(numbered, str(question_number) + option_letter, (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            



    answers = []
    answers_contours = []


    for question_number, row in enumerate(question_grid):
        current_row_levels = []
        for option_number, c in enumerate(row):

            mask = np.zeros(doc_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)


            mask = cv2.bitwise_and(doc_thresh, doc_thresh, mask=mask)
            total = cv2.countNonZero(mask)


            current_row_levels.append(total)

        max_value = max(current_row_levels)

        max_index = current_row_levels.index(max_value)

        current_row_levels.pop(max_index)


        average = sum(current_row_levels) / len(current_row_levels)


        (x, y, w, h) = cv2.boundingRect(row[max_index])
        x += int(w / 2)
        y += int(h / 2)


        if(max_value > average * 1.2):
            answers_contours.append(c)
            answers.append({
                'question': question_number + 1,
                'marked': chr(ord('A') +  max_index),
                'x': x,
                'y': y,
        })
        else:
            answers.append({
                'question': question_number + 1,
                'marked': 'N/A',
        
        })

    for answer in answers:
        if 'marked' in answer and answer['marked'] != 'N/A':
            cv2.putText(numbered, 'X', (answer['x'], answer['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 8)
            

    cv2.imwrite("document.jpg", numbered)

    for answer in answers:
        if 'marked' in answer and answer['marked'] != 'N/A':

            cv2.putText(numbered, 'X', (answer['x'], answer['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 8)
    
    # resize image to 600px width
    width = 500
    height = int(numbered.shape[0] * width / numbered.shape[1])
    dim = (width, height)
    numbered = cv2.resize(numbered, dim, interpolation = cv2.INTER_AREA)


    retval, buffer = cv2.imencode('.jpg', numbered, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    jpg_as_text = str(base64.b64encode(buffer))

    return ("data:image/jpeg;base64," + jpg_as_text[2:-1], answers)



    

            





