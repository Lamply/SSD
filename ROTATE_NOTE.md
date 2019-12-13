# Note
I have changed the project to support rotate dataset roughly. It may work to some extent.  

The main idea is to add a custom dataset like `[x, y, w, h, angle, class]`, and a box head generating 5 channels to predict angle of a box.  

I also write a simple prior boxes class to generate fix height/width ratio boxes with 8 type of angle (0, 22.5, 45, ...).  

Utils are changed mostly to support boxes with angle channel.  

In loss class, I addition angles' mse loss to coordinates' smooth l1 loss.  

Evalution method is imcomplete, my implement just ignore the angle.


