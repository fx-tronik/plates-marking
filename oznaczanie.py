import numpy as np
import cv2
import os
import sys
import csv
import glob
import argparse
import pytesseract
from detekcja import predict


pred = predict()
pred.load()
config = ('--oem 1 --psm 7')
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required=False, default=0,
	help="index klatki od ktorej zaczynamy")
ap.add_argument("-z", "--zmiana", default=20,
	help="path to output label binarizer")
ap.add_argument("-f", "--folder", type=str, default="",
	help="nazwa folderu ze zdjeciami")
ap.add_argument("-v", "--video", type=str, default="",
	help="nazwa pliku video z folderu filmy")
args = vars(ap.parse_args())

if(not args["folder"] and not args["video"]):
    print("Brak okreslonego folderu zdjec lub filmu (-f lub -v)")
    sys.exit(0)
if(args["folder"]):
    film_input = False
if(args["video"]):
    film_input = True

ZBIOR_FOLDER = args["folder"] + '/'
video_nazwa = args["video"]
klawisz_page_zmiana = int(args["zmiana"])
index = int(args["index"])
window_size_x = 1600
window_size_y = 1200

INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'

if(film_input):
	cap = cv2.VideoCapture('input/filmy/' + video_nazwa)
	if(not cap.isOpened()):
		print('Nieprawidlowy plik video')
		sys.exit(0)
	ZBIOR_FOLDER = video_nazwa[:-4] + '/'
else:
	image_path = sorted(glob.glob(INPUT_DIR + ZBIOR_FOLDER + '*'))
	if len(image_path) == 0:
	    print("Brak zdjec w folderze " + INPUT_DIR + ZBIOR_FOLDER)
	    sys.exit(0)

IMAGES_DIR = 'output/images/' + ZBIOR_FOLDER
MASKA_DIR = 'output/off_maska/' + ZBIOR_FOLDER
TABLICE_DIR = 'output/tablice/' + ZBIOR_FOLDER
CSV_FILE = 'output/tablice_oznaczone.csv'

rbuttondown = False
mbuttondown = False
zoomed = False
max_liczba_tablic = 5
max_liczba_znakow = 9
tablica_size = (260, 60)
pkt_tablica = np.float32([[0, 0], [tablica_size[0], 0],
                          [tablica_size[0], tablica_size[1]], [0, tablica_size[1]]])

roi = [(0, 0), (0, 0)]
off_roi = [(0, 0), (0, 0)]
pts = [[] for _ in range(max_liczba_tablic)]
tablica_znaki = ['' for _ in range(max_liczba_tablic)]
tablica_znaki_tmp = tablica_znaki
znak = " 0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"
skala = 1.0
numer = 0
color = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

def mouse_clb(event, x, y, flags, param):
    global roi
    global off_roi
    global rbuttondown
    global mbuttondown
    global pts
    global zoomed
    global skala

    if event == cv2.EVENT_LBUTTONDOWN:
        if rbuttondown == True:
            return

        if(len(pts[numer]) < 4):
            pts[numer].append(
                [int(x/skala + roi[0][0]), int(y/skala + roi[0][1])])
    if event == cv2.EVENT_LBUTTONUP:
        if(len(pts[numer]) == 4):
            M = cv2.getPerspectiveTransform(np.asarray(pts[numer], dtype=np.float32), pkt_tablica)
            dst = cv2.warpPerspective(image, M, (tablica_size[0], tablica_size[1]))
            text = pred.ocr(dst)
            #text = pytesseract.image_to_string(dst, config=config)[1:]    
            tablica_znaki[numer] = text
            

    if event == cv2.EVENT_MBUTTONDOWN:
        mbuttondown = True
        off_roi[0] = off_roi[1] = int(x/skala + roi[0][0]), int(y/skala + roi[0][1])
    if event == cv2.EVENT_MBUTTONUP:
        mbuttondown = False
        off_roi[1] = int(x/skala + roi[0][0]), int(y/skala + roi[0][1])
        off_roi = [np.amin(off_roi, axis = 0), np.amax(off_roi, axis = 0)]
        w, h = abs(off_roi[1] - off_roi[0])
        if(h < 5 or w < 5):
            return
        cv2.rectangle(off_maska, tuple(off_roi[0]), tuple(off_roi[1]), (0,0,0), cv2.FILLED)

    if zoomed == False:
        if event == cv2.EVENT_RBUTTONDOWN:
            rbuttondown = True
            roi[0] = roi[1] = int(x/skala), int(y/skala)
        if event == cv2.EVENT_RBUTTONUP:
            rbuttondown = False
            roi[1] = int(max(0, x/skala)), int(max(0, y/skala))
            roi = [np.amin(roi, axis = 0), np.amax(roi, axis = 0)]
            w, h = abs(roi[1] - roi[0])
            if(h < 10 or w < 10):
                return
            zoomed = True
            skala = np.amin([window_size_x/w, window_size_y/h])

    if event == cv2.EVENT_MOUSEMOVE:
        if zoomed == False:
            roi[1] = int(x/skala), int(y/skala)
        off_roi[1] = int(x/skala + roi[0][0]), int(y/skala + roi[0][1])


def wczytaj_dane(index):
    global pts
    global tablica_znaki
    global numer
    global zoomed
    global roi
    global off_maska
    global image
    global skala

    numer = 0
    zoomed = False
    roi = [(0,0), (0,0)]
    if(film_input):
        cap.set(1, index)
        _, image = cap.read()
        file_folder_name = ZBIOR_FOLDER + str(index) + '.jpg'
    else:
        image = cv2.imread(image_path[index], 1)
        file_folder_name = image_path[index][len(INPUT_DIR):]
    file_name = file_folder_name.split('/')[-1]
    skala = np.amin([window_size_x/image.shape[1], window_size_y/image.shape[0]])

    plik = open(CSV_FILE, 'r', encoding='utf-8-sig')
    CSVreader = csv.reader(plik, delimiter='\t', quotechar='|')
    wiersz = list(CSVreader)

    pts = [[] for _ in range(max_liczba_tablic)]
    tablica_znaki = ['' for _ in range(max_liczba_tablic)]
    for col in wiersz:
        if(file_folder_name == col[0]):                   	# wykryty istniejacy wpis - zdjecia lub film
            znaki = col[1].split(';')        				# odczyt znakow tablicy
            tablica_znaki = ['' for _ in range(max_liczba_tablic)]
            tablica_znaki[:len(znaki)] = znaki[:max_liczba_tablic]
            off_maska = cv2.imread(MASKA_DIR + col[0].split('/')[1][:-3] + 'png', 0)
            if(off_maska is None):
                print("Nieprawidlowa sciezka dla maski: ", file_folder_name)
            for y in range(2, len(col)):
                p = col[y].split(';')
                pts[y-2] = [list(map(int, x.split(','))) for x in p if not x == '']
            break
    else:
        off_maska = np.ones(image.shape[:2], dtype='uint8')

if not os.path.exists(INPUT_DIR):
    print("Brak folderu wejsciowego ", INPUT_DIR)
    sys.exit(0)
if not os.path.exists(OUTPUT_DIR):
    print("Tworzenie folderu ", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(IMAGES_DIR):
    print("Tworzenie folderu ", IMAGES_DIR )
    os.makedirs(IMAGES_DIR)
if not os.path.exists(MASKA_DIR):
    print("Tworzenie folderu ", MASKA_DIR)
    os.makedirs(MASKA_DIR)
if not os.path.exists(TABLICE_DIR):
    print("Tworzenie folderu ", TABLICE_DIR)
    os.makedirs(TABLICE_DIR)

plik = object()
try:
    plik = open(CSV_FILE, 'a+', encoding='utf-8-sig')
except FileNotFoundError:
    print('Plik csv nie istnieje - tworzenie pliku', CSV_FILE)
else:
    print('Wczytano plik ', CSV_FILE)
    plik.seek(0)
    print("Liczba oznaczonych zdjec: ", sum(1 for row in plik))
plik.close()

wczytaj_dane(index)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('image', mouse_clb)


while (1):
    image_display = image.copy()

    if(rbuttondown == True):
        cv2.rectangle(image_display, tuple(roi[0]), tuple(roi[1]), (255,0,0), 2)
    if(mbuttondown == True):
        cv2.rectangle(image_display, tuple(off_roi[0]), tuple(off_roi[1]), (255,255,255), 2)

    image_display = cv2.bitwise_and(image_display, image_display, mask=off_maska)

    for n, p in enumerate(pts):
        if(n >= 4):
            n = 4
        for i in range(1, len(p)):
            cv2.line(image_display, tuple(p[i]), tuple(p[i-1]), color[n], 2)
        if len(p) == 4:
            cv2.line(image_display, tuple(p[3]), tuple(p[0]), color[n], 2)

    if(zoomed == True):
        image_display = image_display[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

    image_display = cv2.resize(image_display, (0, 0), fx=skala, fy=skala)

    cv2.putText(image_display, 'Indeks: ' + str(index), (0,25), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    #cv2.putText(image_display, 'Zbior: ' + ZBIOR_FOLDER[:-1], (0,50), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    cv2.putText(image_display, 'Tablica: ' + tablica_znaki[numer], (0,75), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    #cv2.putText(image_display, 'N: ' + str(liczba_detekcji), (0,100), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)
    #cv2.putText(image_display, 'Skala: ' + "%.2f" % skala, (0,100), cv2. FONT_HERSHEY_PLAIN, 2, color[numer], 2)

    cv2.imshow('image', image_display)

    c = cv2.waitKey(50) % 255

    if(c == -1):
        continue

    if chr(c) in znak.lower():      # zapis wprowadzanych znaków
        if(len(tablica_znaki[numer]) < max_liczba_znakow):
            tablica_znaki[numer] += chr(c).upper()

    if(c >= 176 and c <= 185):      # numpad 0-9
        if(len(tablica_znaki[numer]) < max_liczba_znakow):
            tablica_znaki[numer] += znak[c-175]

    if(c == 27):    # ESC
        break
    if(c == 225 or c == 226):       # SHIFT lewy lub prawy
        pts[numer] = []

    if(c == 9):     # TAB
        tablica_znaki[numer] = tablica_znaki_tmp[numer]

    if(c == 227):    # CTRL lewy
        zoomed = False
        roi = [(0,0), (0,0)]
        skala = np.amin([window_size_x/image.shape[1], window_size_y/image.shape[0]])

    if(c == 8):     # BACKSPACE
        tablica_znaki[numer] = tablica_znaki[numer][:-1]
        
    if(c == 13 or c == 10):    # ENTER
        if(film_input):
            file_folder_name = ZBIOR_FOLDER + str(index) + '.jpg'
        else:
            file_folder_name = image_path[index][len(INPUT_DIR):]
        str_write = file_folder_name + '\t'     									# kolumna 0 - zapis nazwy pliku
        str_write += ";".join([t for t in tablica_znaki if not t == '']) + '\t'     # kolumna 1 - zapis opisanych znakow tablic
        zapis = False
        for i in range(max_liczba_tablic):
            if(tablica_znaki[i] != '' or len(pts[i]) == 4):
                zapis = True
                tablica_znaki_tmp = tablica_znaki
                break

        for i, p in enumerate(pts):
            if(len(p) != 4):
                continue
            str_write += ';'.join([','.join(map(str, x)) for x in p]) + '\t'        # kolumny 2+ zapis 4 punktów w każdej kolejnej kolumnie

            if(tablica_znaki[i] == ''):
                continue
            M = cv2.getPerspectiveTransform(np.asarray(p, dtype=np.float32), pkt_tablica)
            dst = cv2.warpPerspective(image, M, (tablica_size[0], tablica_size[1]))
            cv2.imwrite(TABLICE_DIR + tablica_znaki[i] + '.jpg', dst)

        with open(CSV_FILE, 'r') as csvfile:
            csvReader = csv.reader(csvfile)
            clean_rows = [row for row in csvReader if row[0].split('\t')[0] != file_folder_name]

        with open(CSV_FILE, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(clean_rows)
            if(zapis == True):
                file_name = file_folder_name.split('/')[-1]
                csvfile.write(str_write[:-1] + '\n')                        # zapis bez ostatniego tabulatora
                cv2.imwrite(IMAGES_DIR + file_name, image)                  # oryginalny obraz
                cv2.imwrite(MASKA_DIR + file_name[:-3] + 'png', off_maska)  # maska dla loss weights

        if(film_input):
            c = 85  # przesuniecie filmu
        else:
            c = 83 # przewiniecie zdjecia w prawo

    if(c == 0):     # DEL
        if(film_input):
            file_folder_name = ZBIOR_FOLDER + str(index) + '.jpg'
        else:
            file_folder_name = image_path[index][len(INPUT_DIR):]

        pts = [[] for _ in range(max_liczba_tablic)]
        tablica_znaki = ['' for _ in range(max_liczba_tablic)]
        off_maska = np.ones(image.shape[:2], dtype='uint8')

        with open(CSV_FILE, 'r') as csvfile:
            csvReader = csv.reader(csvfile)
            clean_rows = [row for row in csvReader if row[0].split('\t')[0] != file_folder_name]

        with open(CSV_FILE, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(clean_rows)

        file_name = file_folder_name.split('/')[-1]
        if os.path.exists(IMAGES_DIR + file_name):
            os.remove(IMAGES_DIR + file_name)
        if os.path.exists(MASKA_DIR + file_name[:-3] + 'png'):
            os.remove(MASKA_DIR + file_name[:-3] + 'png')

    if(c == 82):     # strzalka w gore
        if(numer+1 < max_liczba_tablic):
            numer += 1

    if(c == 84):    # strzalka w dol
        if(numer > 0):
            numer -= 1

    if(c == 81 and index > 0):                                          # strzalka w lewo
        index -= 1
        wczytaj_dane(index)
    if(c == 83 and (film_input == True or index < len(image_path)-1)):  # strzalka w prawo
        index += 1
        wczytaj_dane(index)
    if((c==154 or c == 85) and film_input == True):                                 # page-up
        index += klawisz_page_zmiana
        wczytaj_dane(index)
    if((c == 155 or c == 86) and film_input == True and index > klawisz_page_zmiana): # page-down
        index-= klawisz_page_zmiana
        wczytaj_dane(index)

cv2.destroyAllWindows()
