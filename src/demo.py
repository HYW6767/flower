
from flower import FlowerPredictor
from PIL import Image
flower=FlowerPredictor("src/flower.pt")

img1=Image.open("docs/3734.jpg_wh860.jpg")
img2=Image.open("docs/33331825_155457689128_2.jpg")
result,_=flower.pred(img1)
print(f"{img1}是{result}")
result,_=flower.pred(img2)
print(f"{img2}是{result}")