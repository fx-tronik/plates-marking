import numpy as np
import cv2
import os
import random
import math
import sys
import csv
import glob
# test
def mouse_clb(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if rbuttondown == True:
            return
        if(len(pts[numer]) < 4):
            global pts
            pts[numer].append([int(x/skala + roi[0][0]), int(y/skala + roi[0][1])])
    
    global zoomed
    if zoomed == True:
        return
    global roi
    global rbuttondown
    global przesuniecie
    global skala
    
    if event == cv2.EVENT_RBUTTONDOWN:
        rbuttondown = True
        roi[0] = int(x/skala), int(y/skala)
        roi[1] = roi[0]
    if event == cv2.EVENT_RBUTTONUP:
        rbuttondown = False
        roi[1] = int(x/skala), int(y/skala)
        roi = [np.amin(roi, axis = 0), np.amax(roi, axis = 0)]
        w, h = abs(roi[1] - roi[0])
        print(h, w)

        if(h < 10 or w < 10):
            return

        skala = np.amin([window_size_x/w/skala, window_size_y/h/skala])
        zoomed = True
    if event == cv2.EVENT_MOUSEMOVE:
        roi[1] = int(x/skala), int(y/skala)

INPUT_DIR = 'result/'
OUTPUT_DIR = 'output/'
CSV_DIR = 'tablice_oznaczone.csv'

rbuttondown = False
zoomed = False
window_size_x = 800
window_size_y = 600
max_liczba_tablic = 5
max_liczba_znakow = 9
    
roi = [(0,0), (0,0)]
pts = [[] for _ in range(max_liczba_tablic)]
tablica_znaki = ['' for _ in range(max_liczba_tablic)]
znak = " 0123456789ABCDEFGHIJKLMNOPRSTUWXYZ"   
skala = 1.0
numer = 0
color = [(0,0,255), (0,255,0), (255,0,255), (0,255,255), (255,255,255)] 

plik = object()
index = 0
try:
    plik = open(CSV_DIR, 'a+', encoding='utf-8-sig')
except FileNotFoundError:
    print('Wskazany plik nie istnieje... Tworzenie nowego')
else:
    print('Wczytano plik')
    plik.seek(0)
    index = sum(1 for row in plik)
    plik.seek(0,2)
print("Liczba oznaczonych zdjec: ", index)

image_path = glob.glob(INPUT_DIR + '*')
image = cv2.imread(image_path[index], 1)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('image',mouse_clb)

while (1):
    image_display = image.copy()
    if(zoomed == False):
        rows, cols = image_display.shape[:2]
        skala = np.amin([window_size_x/cols, window_size_y/rows])
    
    if(rbuttondown == True):
        cv2.rectangle(image_display, tuple(roi[0]), tuple(roi[1]), (255,0,0), 2)

    for n, p in enumerate(pts):
        for i in range(1, len(p)):
            cv2.line(image_display, tuple(p[i]),tuple(p[i-1]), color[n], 2)
        if len(p) == 4:
            cv2.line(image_display, tuple(p[3]),tuple(p[0]), color[n], 2)
    
    if(zoomed == True):
        image_display = image_display[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        
    image_display = cv2.resize(image_display, (0,0), fx = skala, fy = skala)
    
    cv2.putText(image_display, 'Numer: ' + str(numer+1), (0,25), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    cv2.putText(image_display, 'Znaki: ' + tablica_znaki[numer], (0,50), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    cv2.putText(image_display, 'Skala: ' + "%.2f" % skala, (0,75), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    
    cv2.imshow('image', image_display)
    
    c = cv2.waitKey(50)
    
    if(c == -1):
        continue
    if chr(c) in znak.lower():
        if(len(tablica_znaki[numer]) < max_liczba_znakow):
            tablica_znaki[numer] += chr(c).upper()

    if(c == 27):    # ESC
        break
    if(c == 225 or c == 226):    # SHIFT lewy lub prawy
        image = cv2.imread(image_path[index], 1)
        pts[numer] = []
    if(c == 227):     # CTRL
        zoomed = False
        roi = [(0,0), (0,0)]

    if(c == 8):     # BACKSPACE
        tablica_znaki[numer] = tablica_znaki[numer][:-1]
        image = cv2.imread(image_path[index], 1)
    if(c == 13):    # ENTER
        file_name = image_path[index][len(INPUT_DIR):]
        str_write = file_name + '\t'
        
        for n in tablica_znaki:
            if n == '':
                continue
            str_write += n + ';'
        if(str_write[-1:] == ';'):
            str_write = str_write[:-1]
            
        str_write += '\t'
        
        for p in pts:
            if(len(p) != 4):
                continue
            for k in range(4):                         # współrzędne tablicy rejestracyjnej
                str_write += str(p[k][0]) + ',' + str(p[k][1]) + ';'
            str_write = str_write[:-1] + '\t'
        
        plik.write(str_write[:-1] + "\n")               # zapis bez tabulatora na końcu
        cv2.imwrite(OUTPUT_DIR + file_name, image) 
    
        index += 1
        numer = 0
        zoomed = False
        roi = [(0,0), (0,0)]
        pts = [[] for _ in range(max_liczba_tablic)]
        tablica_znaki = ['' for _ in range(max_liczba_tablic)]
        image = cv2.imread(image_path[index], 1)
    if(c == 82):     # strzalka w gore
        if(numer+1 < max_liczba_tablic):
            numer += 1
        #image = cv2.imread(image_path[index], 1)
    if(c == 84):    # strzalka w dol
        if(numer > 0):
            numer -= 1
        #image = cv2.imread(image_path[index], 1)

plik.close()
cv2.destroyAllWindows()