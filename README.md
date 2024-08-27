# GAME DESCRIPTION
This is Handymath, a simple but fun game involving *hands* and *math*!
This game is suitable for users of all ages, especially young children who are new to the world of math. 
In order to excel in this game, users need to react to newly generated math problems by performing quick mental math.
We aim to provide an interactive learning experience from which children can hone their arithmetic skills. Playing this game regularly can yield positive results in
developing the children's brains and stimulating neural pathways from an early age. We hope that children can not only have fun, but also discover their interest in math through this game.


### How the game works
- The game runs in an infinite loop, randomly generating simple math problems of addition, subtraction, or multiplication.
- The user can solve the problems by holding up hand gestures of numbers 1-9, one digit at a time.
- An image is captured and run through a machine learning model. The model's output is processed to predict the digit that the user is holding up.
- A predicted digit is saved if it matches the answer. AFterwards, the game provides feedback on the correct partial answer.
- If the full answer is correct, the score is recorded, and a new question is generated.


### How we created this
- Take 100-150 photos of hand gestures 1-9, at different angles.
- Train the AI model.

Examples of our photos:

<img src="https://github.com/user-attachments/assets/664d47ab-0bda-4eb1-bd4c-6c7cbfe0e5d6" width="300" />

*Hand gesture of 1.*

 
<img src="https://github.com/user-attachments/assets/02b1198c-f7dc-4eb8-a916-42833fce6f5e" width="300" />
 
*Hand gesture of 2.*





# INSTALLATION INSTRUCTIONS

Connect to the IP address 192.168.55.1:8888.

Enter the password: jetson
 
Install the following:
- trt_pose https://github.com/NVIDIA-AI-IOT/trt_pose.git
- trt_pose_hand https://github.com/NVIDIA-AI-IOT/trt_pose_hand.git
- jetson_dlinano https://github.com/sangyy/jetson-dlinano.git
- jetracer https://github.com/NVIDIA-AI-IOT/jetracer.git
- jupyter_clickable_image_widget https://github.com/jaybdub/jupyter_clickable_image_widget.git
- jetcam https://github.com/NVIDIA-AI-IOT/jetcam.git






