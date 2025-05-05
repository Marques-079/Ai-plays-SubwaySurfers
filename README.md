# Subway Surfers vs AI 🚃 🚃🏃‍♂️ (🚧)

After being inspired by Youtubers making “AI beats…” content I figured why not make one to beat the hit game Subway Surfers! A lot of thinking and planning went into this project, mainly it consists of an ensemble of models to auto-label, analyse and plan our approach to this challenging environment.

# How the model works⁉️

**Data pipeline:** Firstly we must extract key frames from videos and label them for the transformer model, my goal is to label around 1 million+ frames for training data. This is achieved through TWO computer vision models fine tuned off Ultralytics open source models. 

These models identify obstacles and our play and thus, act as the eyes and senses to our AI. We label using a handwritten script which uses the collected obstacles to plot the greedy best path to take in the frame’s instance. This way we can autolabel thousands of frames relatively quickly (faster than me sitting there all day)

<p align="center">
  <img src="https://raw.githubusercontent.com/Marques-079/Ai-plays-SubwaySurfers/30665ba9d212de2fdb5ce993e7af52ea215c47f2/Images/2025-05-03.jpg"
       width="500"  alt="gif1">
</p>

**Dual models :**  I run real time analysis on our gameplay using a screen overlay, essentially while this runs I have a Convolutional Neural Network (CNN) which acts as an encoder for our transformer model. To train we use our 1 million+ frames (run through CNN) and then passed to our transformer. The goal here is to establish a baseline adaptable model that not only uses context of 16 frames but also can string together combos (greedy could not). 

A consideration here was keeping inference less than 200ms per frame. Therefore we can run updates to our model 5 times per second, slightly faster than human reaction time, except we get a much higher movement accuracy in the moment. 

<p align="center">
  <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/raw/dee3a84cb4a8d705b0de6bb666cd7cf2bf6880ea/Images/Screenshot%202025-05-041.jpg"
       width="45%"  alt="gif1">
</p>

**PPO and Fine tuning:** To reach a 100% accuracy I plan to fine tune the transformer model (CNN frozen weights to prevent model instability)  using reinforcement learning. Ideally our pre-trained transformer is accurate enough to have long runs without mistakes! 

<br>

---


## Step 1  - Computer Vision and ground detection 👀

My first thought was to quantise the image into smaller pixels and through this I could run a model to detect obstacles based on colour, little did I know at the time this would be a HUGE under estimation of the task at hand

<table>
  <tr>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Marques-079/Ai-plays-SubwaySurfers/31d319bc0a0c6f48d1f8a93a5de47754fbc022c4/Images/Screenshot121212.png"
           alt="Pixel‑palette rail mask"
           style="width:100%;"/>
    </td>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Marques-079/Ai-plays-SubwaySurfers/7f3f07a7a62fa7c48c1a61735864e986e409dcff/Images/Screenshot%202025-05-034.jpg"
           alt="3D‑LUT recoloured frame"
           style="width:100%;"/>
    </td>
  </tr>
</table>



Initial thoughts on how we could detect obstacles. 

<br>

---


I was soon quick to realise that this would not work at all, fine details were missed and there was too much variance in the consistent game obstacles due to warping and different perspectives. Clearly I needed something smarter which is how I arrived at a computer vision modes, one to detect Jake (the runner) and one to detect obstacles in the image. 

And thus came the merticlous task of labelling data for the computer vision model to train on. I spent around 5 hours labelling for the 2 models, but it was worth it in the end!


<table>
  <tr>
    <td>
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/raw/30665ba9d212de2fdb5ce993e7af52ea215c47f2/Images/2025-05-03%20at.jpg" width="420"/>
    </td>
    <td rowspan="2">
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/blob/cdeb34fdc1248284687e4e1af8f4985e3d17fa77/Images/inclasssubwaysurfers.jpg" width="460"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/blob/4cdecb4ba31c35b715c30faf47c2d0fe80205bce/Images/Screenshot%20Masks11.png" width="420"/>
    </td>
  </tr>
</table>


Hand cramps were real… 

The reason why this took to much time is because a model can only predict as well as the data that trained it, any mistakes I made (especially repetatively) would throw off the model's predictive ability by a large amount - so it was key here to fine-tune it on highly accurate data. 

<br>

---


Using Roboflow's built in user interface made fine tuning these open source models was drastically sped up (Thanks roboflow!). Through a bit of experimentation, early stopping and late nights here were the performance of my two models - the mAP@50 scores are a bit lower than what the actual model detects due to me making a few mistakes in data labelling. In the end I used around 10000 labelled images to fine tune - this number is post augmentation. For obstacles I used horizontal mirroring and varied zoom 0% → 18% and for Jake I used image rescaling and horizontal mirror.

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/raw/a733c5387799bebe6849329cc0c3ab3a24b0dd42/Images/Screenshot%202025-05-032.jpg"
           alt="Pixel‑palette rail mask"
           style="width:100%;"/>
    </td>
    <td width="50%">
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/raw/a733c5387799bebe6849329cc0c3ab3a24b0dd42/Images/Screenshot%202025-05-033.jpg"
           alt="3D‑LUT recoloured frame"
           style="width:100%;"/>
    </td>
  </tr>
</table>

- Images from Roboflow’s dashboard

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/blob/a733c5387799bebe6849329cc0c3ab3a24b0dd42/Images/Screenshot%202025-05-036.jpg"
           alt="Pixel‑palette rail mask"
           style="width:100%;"/>
    </td>
    <td width="50%">
      <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/blob/a733c5387799bebe6849329cc0c3ab3a24b0dd42/Images/Screenshot%202025-05-037.jpg"
           alt="3D‑LUT recoloured frame"
           style="width:100%;"/>
    </td>
  </tr>
</table>

Computer vision - Mk1 at work 

- Heatmap of annotations in data (pretty accurate to real game) Credit: Roboflow dashboard

The decision to use this method instead of hand labelling was driven by the fact that there is many hours of subway surfers footage online in version 1.0, instead of playing I can just take screenshots from those videos and run analysis on them - skill is not so much of an issue as we are looking for scenarios rather than excellent gameplay (another benefit of this approach)

<br>

---


# The greedy algorithm  👹

Now comes arguably the hardest part of this process, making an auto labeller which runs at “efficient speeds” of under 500ms per image and maintains an exceptionally high accuracy in greedy decision making. In my mind there was two game states I had to code for. Jake in the Air and Jake on the ground. On the ground he should play obstacle avoidance and stay on rails (if not on rails he would have crashed), and if on trains we should aim to jump from train to train. Also implementing a movement cooldown after a move would be paramount to eliminating sudden erros.

My conclusion was an algorithm which determines if he is Air or Ground based off height, this can be paired with a standing on algorithm to yield accurate results. When working on rails we would also have to be able to identify possible moves. How the greedy algorithm works for game state: ground is that it tracks forward the rails it has identified, and then attempts to move into the one with the highest longevity (ignore no kill obstacles eg.  low barriers). 

<table>
  <tr>
    <td>
      <img src="Images/Screenshot%202025-05-038.jpg" width="420"/>
    </td>
    <td rowspan="2">
      <img src="Images/Screenshot%202025-05-039.jpg" width="460"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="Images/Screenshot%202025-05-040.jpg" width="420"/>
    </td>
  </tr>
</table>



- Bottom left image says “Jake is on the ground” - Airtime algorithm

Valid ground pathing is found by combining matching colours within the identified mask for rails this prevents colours that match rails, but aren’t rails to being identified. The downside is that if our algorithm fails then we will be running blind. This reliance could be a bad thing. 

<br>

---


# CNN and Transformer training 🚗🔄🤖

🚧 (Project in progress) 🚧

---

# Proximal Policy Optimisation  🤓

🚧 (Project in progress) 🚧

---

# Terminus.

🚧 (Project in progress) 🚧
