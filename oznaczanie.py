import numpy as np
import cv2
import os
import sys
import csv
import glob

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
        if(h < 10 or w < 10):
            return

        skala = np.amin([window_size_x/w/skala, window_size_y/h/skala])
        zoomed = True
    if event == cv2.EVENT_MOUSEMOVE:
        roi[1] = int(x/skala), int(y/skala)

def wczytaj_dane(f_name):
    plik = open(CSV_FILE, 'r', encoding='utf-8-sig')
    CSVreader = csv.reader(plik, delimiter='\t', quotechar='|')
    wiersz = list(CSVreader)
        
    global pts
    global tablica_znaki
        
    for col in wiersz:
        if(f_name == col[0]):                # wykryty istniejacy wpis
            tablica_znaki = col[1].split(';')   # odczyt znakow tablicy
            pts = []
            for y in range(2, len(col)):
                p = col[y].split(';')
                pts.append(np.array([x.split(',') for x in p], dtype=np.int32))
            if(len(pts) == 0):
                pts.append([])
            break       
        else:
            pts = [[]]
            tablica_znaki = [f_name]
    else:
        pts = [[]]
        tablica_znaki = [f_name]

INPUT_DIR = 'input/'
TABLICE_DIR = 'tablice/'
CSV_FILE = 'tablice_oznaczone.csv'
CURRENT_DIR = os.getcwd()

rbuttondown = False
zoomed = False
window_size_x = 800
window_size_y = 600
max_liczba_tablic = 5
max_liczba_znakow = 9
tablica_size = (260, 60)
pkt_tablica = np.float32([[0,0],[tablica_size[0],0], 
                          [tablica_size[0],tablica_size[1]], [0, tablica_size[1]]])

roi = [(0,0), (0,0)]
pts = [[] for _ in range(max_liczba_tablic)]
tablica_znaki = ['' for _ in range(max_liczba_tablic)]
znak = " 0123456789ABCDEFGHIJKLMNOPRSTUWXYZ"   
skala = 1.0
numer = 0
color = [(0,0,255), (0,255,0), (255,0,255), (0,255,255), (255,255,255)] 

if not os.path.exists(CURRENT_DIR + '/' + INPUT_DIR):
    print("Brak zbioru wejściowego")
    sys.exit(0)
if not os.path.exists(CURRENT_DIR + '/' + TABLICE_DIR):
    print("Brak folderu dla tablic - tworzenie folderu ", TABLICE_DIR[:-1])
    os.makedirs(CURRENT_DIR + '/' + TABLICE_DIR)

plik = object()
index = 0
try:
    plik = open(CSV_FILE, 'a+', encoding='utf-8-sig')
except FileNotFoundError:
    print('Plik csv nie istnieje - tworzenie pliku', CSV_FILE)
else:
    print('Wczytano plik ', CSV_FILE)
    plik.seek(0)
    index = sum(1 for row in plik)
print("Liczba oznaczonych zdjec: ", index)
plik.close()


image_path = glob.glob(INPUT_DIR + '*')
if index >= len(image_path):
   index = len(image_path)-1 
file_name = image_path[index][len(INPUT_DIR):]
image = cv2.imread(image_path[index], 1)
wczytaj_dane(file_name)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('image', mouse_clb)

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
    
    cv2.putText(image_display, 'Numer: ' + str(index+1), (0,25), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    cv2.putText(image_display, 'Tablica: ' + tablica_znaki[numer], (0,50), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
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
        pts[numer] = []
    
    if(c == 227):     # CTRL
        zoomed = False
        roi = [(0,0), (0,0)]

    if(c == 8):     # BACKSPACE
        tablica_znaki[numer] = tablica_znaki[numer][:-1]
    
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
        
        for i, p in enumerate(pts):
            if(len(p) != 4):
                continue
            for k in range(4):            # współrzędne tablicy rejestracyjnej
                str_write += str(p[k][0]) + ',' + str(p[k][1]) + ';'
            str_write = str_write[:-1] + '\t'

            M = cv2.getPerspectiveTransform(np.asarray(p, dtype=np.float32), pkt_tablica)
            dst = cv2.warpPerspective(image, M, (tablica_size[0], tablica_size[1]))
            cv2.imwrite(CURRENT_DIR + '/' + TABLICE_DIR + tablica_znaki[i] + '.jpg', dst)
        
        str_write = str_write[:-1] + "\n"   # przygotowany wiersz do zapisu
        
        with open(CSV_FILE, 'r') as csvfile:
            csvReader = csv.reader(csvfile)
            clean_rows = [row for row in csvReader if (row[0].split('\t'))[0] != file_name]

        with open(CSV_FILE, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(clean_rows)
            csvfile.write(str_write)     
         
        c = 83 # przewiniecie zdjecia w prawo
    

    if(c == 82):     # strzalka w gore
        if(numer+1 < max_liczba_tablic):
            numer += 1
            if(numer >= len(tablica_znaki)):
                tablica_znaki.append('')
            if(numer >= len(pts)):
                pts.append([])
    if(c == 84):    # strzalka w dol
        if(numer > 0):
            numer -= 1
            
    if(c in [81, 83]):          # jesli strzalka w lewo lub w prawo
        tmp_index = index
        if(c == 81 and index > 0):    # strzalka w lewo
            index -= 1
        if(c == 83 and index < len(image_path)-1):    # strzalka w prawo
            index += 1
        if(tmp_index != index):     # jesli byla zmiana zdjecia
            numer = 0
            zoomed = False
            roi = [(0,0), (0,0)]
            image = cv2.imread(image_path[index], 1)
            file_name = image_path[index][len(INPUT_DIR):]
            
            wczytaj_dane(file_name)

cv2.destroyAllWindows()