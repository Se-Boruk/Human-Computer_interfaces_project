import cv2
import mediapipe as mp
import mouse

#Pobranie klasyfikatora
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Element kodu odpowiedzialny za wykrywanie dloni (max 2 dlonie,ufnosc)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

#Wybrana czcionka
font = cv2.FONT_HERSHEY_COMPLEX

state=False #Zmienna sprawdzajaca czy kliknieto mysz
once_in_box=True # Zmienna sprawdzająca czy ręka znalazła się w okreslonej czesci obrazu

################################
#Przechwytywanie kamery internetowej
################################

#Utworzenie obiektu video
vid = cv2.VideoCapture(0)

#Stworzenie petli 
while(True):
      
    #Utworzenie okna, tryb pełnoekranowy
    cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
    #Przechwycenie obecnej klatki z kamery
    frame=vid.read()[1]
    #Stworzenie odbicia lustrzanego przechwytywanego obrazu (aby bylo widac sie na ekranie jak w lustrze)
    frame=cv2.flip(frame,1)
    



    ############################
    #Nalozenie siatki (Gora,dol,prawo,lewo) 
    
    
    #Mapowanie wysokosci i szerokosci klatki
    hm, wm, cm = frame.shape
    

    #horizontal_line
    for i in range(wm):
        for k in range(cm):
            frame[round(hm/2),i,k]=255

    #vertical_line
    for i in range(hm):
        for k in range(cm):
            frame[i,round(wm/2),k]=255
                        
    #Wyrysowanie mapy sterowania myszka
    #horizontal_lines
    for i in range(wm):
        for k in range(cm):
            if k==0:
                frame[round(hm/3),i,k]=0
            else:
                frame[round(hm/3),i,k]=255
                
    for i in range(wm):
        for k in range(cm):
            if k==0:
                frame[round(hm/1.5),i,k]=0
            else:
                frame[round(hm/1.5),i,k]=255
                
                
    #vertical_lines
    for i in range(hm):
        for k in range(cm):
            if k==0:
                frame[i,round(wm/3),k]=0
            else:
                frame[i,round(wm/3),k]=255
                
    for i in range(hm):
        for k in range(cm):
            if k==0:
                frame[i,round(wm/1.5),k]=0
            else:
                frame[i,round(wm/1.5),k]=255

        
    ############################
    #Wykrywanie twarzy
    #####################################################
    #Konwersja do skali szarosci, aby klasyfikator mogl wykrywac twarze
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = face_detector.detectMultiScale(frame_grayscale , 1.05 , 4)
    
    
    # Nałożenie obrysów twarzy na oryginalną klatkę w zaleznosci od ilosci wykrytych obiektow (ilosc twarzy)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   
        #Okreslenie koordynatow srodka twarzy, sprawdzenie w jakiej czesci ekranu sie znajduje sie znajduje
        middle_x=x+0.5*w
        middle_y=y+0.5*h
        
        if middle_y<round(int(hm)/2):
            text1="Gorny "

        else:
            text1="Dolny "

            
        
        if middle_x>round(int(wm)/2):
            text2="prawy "

        else:
            text2="lewy "

            
        #Pokazanie tekstu okreslonego wczesniej

        # Po tekscie: koordynaty, czcionka, rozmiar, kolor, grubosc
        cv2.putText(frame,text1+text2,(x,y-10),font,1,(0,255,0),1)
        
    #####################################################

                
                
    #Poruszanie myszką
    #Gora
    try:
        if middle_y<hm/3 and state==True:
            mouse.move(x=0,y=-5,absolute=False)
            
        #Dol
        if middle_y>hm/1.5 and state==True:
            mouse.move(x=0,y=5,absolute=False) 
            
        #Lewo
        if middle_x<wm/3 and state==True:
            mouse.move(x=-5,y=0,absolute=False) 
            
        #Prawo
        if middle_x>wm/1.5 and state==True:
            mouse.move(x=5,y=0,absolute=False) 
    except:
        #print("nie wykryto twarzy")
        #Nie wykryto twarzy
        pass
        
    
        
    
        
                

    
    
    

    
        
    ##########################################    
    #Wykrywanie dłoni w gornej czesci klatki, zebranie koordynatow jointow do utworzenia boxa
    
    coord_list_x=[]
    coord_list_y=[]
    
    frame_hand = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_hand)
    

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x *wm), int(lm.y*hm)
                coord_list_x.append(cx)
                coord_list_y.append(cy)
                
                x_min=min(coord_list_x)
                y_min=min(coord_list_y)
                x_max=max(coord_list_x)
                y_max=max(coord_list_y)
                hand_center=(round((x_min+x_max)/2),round((y_min+y_max)/2))

                
            if hand_center[0]>wm/2:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)    
                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max)  , (0, 0, 255), 2)   
                
            if hand_center[0]>wm/2 and hand_center[1]<hm/2:
                cv2.putText(frame,"Klikniecie myszy",(wm-290,30),font,1,(0,0,255),1)
                cv2.putText(frame,"Mysz: "+str(state),(wm-290,55),font,1,(255,255,255),1)
                
                
                #Fragment odpowiadajacy za zalaczenie i wylaczenie myszy
                
                if state==False and once_in_box==True:
                    state=True
                    once_in_box=False
                    mouse.click()
                    
                elif state==True and once_in_box==True:  
                    state=False
                    once_in_box=False


                                

                
            else:
                cv2.putText(frame,"Klikniecie myszy",(wm-290,30),font,1,(255,255,255),1)
                cv2.putText(frame,"Mysz: "+str(state),(wm-290,55),font,1,(255,255,255),1)
                once_in_box=True
    else:
        cv2.putText(frame,"Klikniecie myszy",(wm-290,30),font,1,(255,255,255),1)
        cv2.putText(frame,"Mysz: "+str(state),(wm-290,55),font,1,(255,255,255),1)
        once_in_box=True
        
            
        
    #Pokazanie aktualnej klatki
    cv2.imshow('img', frame) 


    #Przerwanie przechwytywania po nacisnieciu klawisza q
    if cv2.waitKey(1)== ord('q'):
        break
    
#Zamkniecie utworzonego okna 
cv2.destroyAllWindows()

#https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
#https://stackoverflow.com/questions/17696061/how-to-display-a-full-screen-images-with-python2-7-and-opencv2-4
#https://medium.com/analytics-vidhya/image-flipping-and-mirroring-with-numpy-and-opencv-aecc08558679
#https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
#https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
#https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/
#https://www.youtube.com/watch?v=YGla_Is2wdU
#https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
#https://stackoverflow.com/questions/1181464/controlling-mouse-with-python






















