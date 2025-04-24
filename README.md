* Project Name:
-----------------------------------------------------------------------------------------------------------------
- Analyzing Micro-Expressions for Lie Detection using Computer Vision (Truth or Lie)

* Project Objective: 
-----------------------------------------------------------------------------------------------------------------
- Implementing Computer vision techniques to analyze micro-expressions in facial images to effectively 
distinguish between truthful and deceptive behaviors, utilizing the provided dataset.
  
* Project Description:
-----------------------------------------------------------------------------------------------------------------
- This project was carried out as part of the Final Project Evaluation under the Employment Enhancement Program 
  conducted by Cranes Varsity. The initiative aims to bridge the gap between academic learning and industry 
  requirements by providing students with hands-on experience in real-world applications.
  
- As a culminating phase of the program, this project reflects the practical skills, technical knowledge, and 
  problem-solving abilities developed throughout the training. The project not only served as an evaluation tool 
  but also provided an opportunity to implement industry-relevant concepts and best practices in a structured, 
  goal-oriented environment.

* Project background:
-----------------------------------------------------------------------------------------------------------------
- Micro-Expressions can occur either as the result of conscious suppression or unconscious repression of emotions. 
  As such, spotting Micro-Expressions(or analyzing facial expressions) is key to learning how to detect lies, 
  unvelling concealed emotions, and deception.

****** Creation of Micro-Expression Image Dataset ******

- Kaggle Dataset Link: https://www.kaggle.com/datasets/devvratmathur/micro-expression-dataset-for-lie-detection

Micro-expressions are subtle and involuntary facial expressions that reflect the true emotions of an individual. 
Due to the impact areas they are relevant to e.g. psychology, security, and human-computer interaction, research 
of this kind has been somewhat hindered still by a lack of high-quality, detailed datasets. There are currently no 
complete micro-expression datasets on Kaggle or any public repositories. Identifying this gap together with the 
particular needs of the mission led us to the challenge of building a new, detailed image dataset dedicated to 
micro-expressions. This dataset aims to serve as a foundational resource for researchers and developers, 
facilitating advancements in the detection, analysis, and interpretation of micro-expressions.

****** Q & A held in Hinglish (Hindi + English) Language ******

Dimension : 560*560px each image

format : PNG

Nothing Phone (2)
Primary Camera Features - 50 MP(OIS) + 50MP 

Dual Camera Setup: 50MP Main Camera (Sony IMX890 Sensor, f/1.88 Aperture, 1/1.56 inch Sensor Size, 1 um Pixel Size,
Focal Length: 24 mm, OIS and EIS Image Stabilisation, Camera Features: Advanced HDR, Motion Capture 2.0, Night Mode,
Portrait Mode, Motion Photo, Super Res Zoom, Lenticular (Filter), AI Scene Detection, Expert Mode, Panorama, 
Panorama Night Mode, Document Mode) + 50MP Ultra Wide Camera (Samsung JN1 Sensor, f/2.2 Aperture, 1/2.76 inch Sensor Size,
EIS Image Stabilisation, FOV: 114 Degree, Camera Features: Advanced HDR, Night Mode, Motion Photo, Lenticular (Filter), 
Macro (4 cm)

Video Recording Resolution
Rear: 4K (at 60fps), Live HDR (4K (at 30 fps)), 1080p (at 30 fps/60 fps), Slow-Mo (480 fps)| Front: 1080p (60 fps), 
Live HDR (1080p (at 30 fps))

****** Step-by-Step Description of the Project ******

# How to Run the code to Create the model in Jupyter Notebook:

1. Open the Kaggle Link -> Go to Download at the topmost right corner of the webpage -> Click Download Dataset as .zip
2. Extract the downloaded .zip folder and rename it to -> "micro-expression-dataset-for-lie-detection"
3. Move the Renamed folder to the same directory as the project.
4. Now go the cell in Notebook and Run it step by step from the start serially.
5. Finally, a Model with extention (.keras) will be created and saved automatically at the same directory.

* N.B:
- As the Dataset is a large sized File, it is not included in the Folder from the begining. Don't worry it has already been tested.
  So, You have to follow the above steps to run and create the model efficiently. 

# How to run the Streamlit App:

1. First upload your whole dirctory to your github reprository with the help of github desktop app.
2. Open streamlit.app website at your webpage.
3. Your online streamlit.app account should be logged in with the Id of your github account.
4. Click on the create your app in the top right corner of the webpage.
5. Select your reprository where your app.py file should be.
6. Then select app.py file.
7. Click continue/run and it will run your app.py streamlit file in the browser.