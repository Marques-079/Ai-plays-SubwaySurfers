# CURRENTLY IN PROGRESS (ETC: Before next year 🥴)

# Subway Surfers vs AI 🚃 🚃🏃‍♂️ (🚧)

After being inspired by Youtubers making “AI beats…” content I figured why not make one to beat the hit game Subway Surfers! A lot of thinking and planning went into this project, mainly it consists of an ensemble of models to auto-label, analyse and plan our approach to this challenging environment.

#TODO INSERT GIFS

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

### Visualisation of logic network for executing logic 

<p align="center">
  <img src="https://github.com/Marques-079/Ai-plays-SubwaySurfers/blob/59dbedd091266dcb2f6f0e9d392c5f8a6cf963be/Irrelevant_files/Images/Network%20for%20subnway.png"
       width="2000"  alt="gif1">
</p>

This algortihm took around 50% of the total build time of this project. The GOAL here was a hardcoded model plugged into the CV models and other detection systems which can play the game for around 5-10 mins at a time before dying due to the inadaptability of hardcoded programming. A large amount of time was spent on the the pathing and detecion systems; as the game ramps in speed we have lesser and lesser reaction time before collision. It was a MUST to have the fasted code possible which resulted in the following few changes : 

A) Swapped MSS (Multiple screenshots) for a customer Swift script. Took capture time from 40ms -> 1ms
B) Used Parsec mirroing between devices for game emulation. Interaction loop 50ms+ -> 10ms 
C) General optimisation of code for speed and efficiency. 200ms -> 60ms 
D) Togglable frame saves -> 550ms -> 180ms

### This led to an 83% time save for the frame analysis loop

Screenshots below using hardcoded frame analysis. 

#TODO INSERT DEVS IMAGES

The back bone of this Algorithm put VERY simply is, knowing what lane you are in + on ground of ontop of trains (diff logic applied) -> Upcoming obstacle? Set evasive timer -> Move 
---
## 🧭 Frame Analysis — Logic Network

```mermaid
flowchart LR
  %% ---------- Styles ----------
  classDef cap  fill:#0ea5e9,stroke:#0369a1,color:#fff;     %% Capture
  classDef inf  fill:#d946ef,stroke:#a21caf,color:#fff;     %% Inference
  classDef post fill:#f59e0b,stroke:#b45309,color:#111;     %% Postproc (ground)
  classDef dec  fill:#22c55e,stroke:#15803d,color:#111;     %% Decision
  classDef out  fill:#3b82f6,stroke:#1d4ed8,color:#fff;     %% Outputs
  classDef top  fill:#fde68a,stroke:#92400e,color:#111;     %% On-top branch
  classDef aux  fill:#cbd5e1,stroke:#475569,color:#111;     %% Aux/infra
  linkStyle default stroke:#64748b,stroke-width:2px;

  %% ---------- Capture ----------
  subgraph CAPTURE
    cap1["ring_grab → frame_bgr"]:::cap
    cap2["Kill-switch pixel check (mss)"]:::cap
    cap3["Lane detect by whiteness"]:::cap
    cap4["Boot-time save toggle (G / Quartz)"]:::cap
    cap5["Movement mute window (0.5s → mute → unmute)"]:::cap
    cap6["Percent-of-color RGBA gauge"]:::cap
  end

  %% ---------- Inference ----------
  subgraph INFERENCE
    inf1["YOLO(segment).predict"]:::inf
    inf2["Device select (CUDA/MPS/CPU) + half"]:::inf
    inf3["Model fuse + warmup"]:::inf
    inf4["compute_on_top_state_fast()"]:::inf
    inf5["OnTopTracker (train/ramp/rails debounce)"]:::inf
  end

  cap1 --> cap2 --> cap3 --> cap4 --> cap5 --> cap6 --> inf1 --> inf2 --> inf3 --> inf4 --> inf5

  %% ---------- On-top gate ----------
  gate{ON_TOP ?}
  inf5 --> gate

  %% ---------- Ground post-proc ----------
  subgraph GROUND_POSTPROC["GROUND — post-processing"]
    post1["process_frame_post()"]:::post
    post2["Promote lowbarrier (HSV wall)"]:::post
    post3["Rails union → GREEN highlight"]:::post
    post4["Heatmap → purple triangles"]:::post
    post5["Pick by bearing (Jake lane)"]:::post
    post6["Curved rays → hit class & dist"]:::post
    post7["Green→Red relabel (near-Y)"]:::post
    post8["Side→Mid flip (ray-tip distance)"]:::post
    post9["tri_summary"]:::post
  end

  gate -- "No" --> post1
  post1 --> post2 --> post3 --> post4 --> post5 --> post6 --> post7 --> post8 --> post9

  %% ---------- Decision logic ----------
  subgraph DECISION_LOGIC
    dec1["Jake-lane impact timers<br/>(starburst→ray for JUMP/DUCK)"]:::dec
    dec2["Tokenized timer + LUT (px→s)"]:::dec
    dec3["Lateral pathing:<br/>RED → HYPERGREEN → GREEN → far YELLOW → least-bad RED"]:::dec
    dec4["Re-entry ban (hysteresis, windowed)"]:::dec
    dec5["Lane change (left/right) + sidewalk jump→duck"]:::dec
  end

  post9 --> dec1 --> dec2 --> dec3 --> dec4 --> dec5

  %% ---------- Outputs ----------
  subgraph OUTPUTS["Actions & outputs"]
    out1["pyautogui actions"]:::out
    out2["pause_inference / cooldown"]:::out
    out3["Overlay render + selective save"]:::out
    out4["Replay dump (optional)"]:::out
    out5["Timing + micro-prof summary"]:::out
  end

  dec5 --> out1 --> out2 --> out3 --> out4 --> out5

  %% ---------- Top branch ----------
  subgraph TOP_BRANCH["TOP — train/ramp flow"]
    top1["do_top_logic_from_result(TL_modular)"]:::top
    top2["Sticky guard (no drop 0.3s)"]:::top
    top3["Save analysed frame (optional)"]:::top
  end

  gate -- "Yes" --> top1 --> top2 --> top3

  %% ---------- Aux / Infra ----------
  subgraph AUX_INFRA["Aux / infra"]
    aux1["Global mute (--quiet/--silent)"]:::aux
    aux2["Warnings/logging off; dprint lazy debug"]:::aux
    aux3["LUT bump after 60s runtime"]:::aux
    aux4["Resource monitors (CPU/RAM/GPU)"]:::aux
    aux5["G toggle to save overlays"]:::aux
  end

```
---


# CNN and Transformer training 🚗🔄🤖

🚧 (Project in progress) 🚧

---

# Proximal Policy Optimisation  🤓

🚧 (Project in progress) 🚧

---

# Terminus.

🚧 (Project in progress) 🚧
