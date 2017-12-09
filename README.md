# recolor
A small piece of code that recolors an image using the coloring style of another image. 
### Usage:
If I want to recolor the image "logo.png" using the color of the image "plants.jpg", I just need to run the following piece of code:
```python
from recolor import Recolor <br>
c = Recolor('plants.jpg') # train the model to get the coloring style of "plants.jpg" <br>
c.recolor('logo.png')
```

After training for the first time, next time you can just load the saved model:

```python
from recolor import Recolor
c = Recolor('plants-colors.npy') # load the model
c.recolor('logo.png')
```

For demonstrating examples or feedbacks, you may refer to https://zcao.info/2017/12/08/a-simple-code-for-recoloring-an-image-using-the-style-of-a-reference-picture/
