from PIL import Image,ImageFont,ImageDraw
import os
from PIL import Image,ImageFont,ImageDraw
import pandas as pd
import random

data_number = 20
save_path = "./test"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# f = open("./label.txt","w")
for i in range(data_number):
    data = open("./dicts.txt",'r').readlines()[1:]
    data = [i.replace("\n","") for i in data]

    text  = "".join(random.sample(data,random.randint(1,15)))
    font = ImageFont.truetype("./TIMES.TTF", 35)

    im = Image.new("RGB", (int(len(text)*20*0.9),random.randint(43,50)), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((5, 1), text, font=font, fill="#000000")
    im.save(os.path.join(save_path,str(i).zfill(5)+".jpg"))
    # f.write(str(i).zfill(5)+".jpg"+"###"+text+"\n")
# f.close()        
