# style-transfer-vgg16
Tried to implement [Gatys & Others](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf]) paper about transfering image style using CNN.  
The paper uses VGG16 net.




## Example
In this example I transfer the Kanagawa's wave style into a tiger's image. Both taken from Google Images.  

Content Image:  
<img src="input/tiger.jpg"
     alt="Tiger"
     height="200px"
     style="float: left; margin-right: 10px" />  
     
Style Image:  
<img src="input/wave.jpg"
     alt="wave"
     height="200px"
     style="float: left; margin-right: 10px" />   
     
Result:     
<img src="output/styled-tiger.jpg"
     alt="Tiger"
     height="200px"
     style="float: left; margin-right: 10px" /> 
