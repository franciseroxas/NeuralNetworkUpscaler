import cv2
import numpy as np
import os

if(os.path.exists("dataset_big2") == False):
    os.mkdir("dataset_big2")
    os.mkdir("dataset_small2")

for i in range(1, 1005):
    for j in range(1, 19):
        chapterNumber = str(i)
        chapterNumber = chapterNumber.zfill(4)

        pageNumber = str(j)
        pageNumber = pageNumber.zfill(3)

        print(chapterNumber + "/" + str(i)+"-"+str(j)+".png")
        if(os.path.exists(chapterNumber + "/" + str(i)+"-"+str(j)+".png") == False):
            continue
        
        try:
            img = cv2.imread(chapterNumber + "/" + str(i)+"-"+str(j)+".png")
        except:
            continue
                
        for k in range(15):
            randHeight = np.random.randint(0, img.shape[0] - 256)
            randWidth = np.random.randint(0, img.shape[1] - 256)
            
            bigImg = img[randHeight:randHeight+256, randWidth:randWidth+256, :]
            smallImg = cv2.resize(img, dsize = (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv2.INTER_AREA)
            smallImg = smallImg[int(randHeight/4):int((randHeight+256)/4), int(randWidth/4):int((randWidth+256)/4), :]
            cv2.imwrite("dataset_big2/"+chapterNumber+"-"+pageNumber+"-"+str(k)+"_bw.png",
                    cv2.cvtColor(bigImg, cv2.COLOR_BGR2GRAY))
            cv2.imwrite("dataset_small2/"+chapterNumber+"-"+pageNumber+"-"+str(k)+"_bw.png",
                    cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY))
        del img

