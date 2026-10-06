
from main import FlowerPredictor
flower=FlowerPredictor("src/flower.pt")

img1="3734.jpg_wh860.jpg"
img2="33331825_155457689128_2.jpg"
result,_=flower.pred("docs/"+img1)
print(f"{img1}是{result}")
result,_=flower.pred("docs/"+img2)
print(f"{img2}是{result}")