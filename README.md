# DeepTrust-Detecting-Deepfakes
Detecting AI-tampered and generated videos to ensure media authenticity. It uses transfer learning by extracting spatial features from InceptionV3 (pretrained model) and feeds them into a custom LSTM model to capture temporal inconsistencies often found in deepfakes, leveraging LSTM’s short-term memory to analyze frame sequences.
<br>
## Installation and Dependencies
1. **Create a Virtual Environment:** I recommend doing so to avoid inconsistencies between different library versions. Use the command `conda create -n deepfake_env python=3.12.4` to create a virtual env and use `conda activate deepfake_env` to activate it.
2. **Clone the repository:** Use  `git clone _link_` to clone it to your local machine. Move to the repo's directory by using `cd _folder_name_`.
3. **Install all Dependencies:** Use `conda install -r requirements.txt` to install libraries.
4. **Run the Streamlit App:** Use `Streamlit run app.py` to run the app.
## About the Dataset
Dataset can be downloaded from here: [mini_face_forensics](https://www.kaggle.com/datasets/rahulkumarroy92/mini-face-forensics)
<br>
The Dataset contains 200 high-quality real and fake videos in each category.
> [!Note]
> If you wish to recreate your own model, and have resources(a good gpu), you can try with original face forensics++ dataset, and tweak some hyperparameters like increased image resolution of frames extracted or number of frames extracted itself, Or you can do some tweaks in LSTM model structure and its parameters.
## Steps Taken in the Project
1. Data Colleection
2. Loading Data
3. Extracting Frames from Videos
4. Feature Extraction
5. LSTM model building and training
6. Testing
7. Deployment
## Preview
https://github.com/user-attachments/assets/5b3efe6b-319b-4d18-aece-66e432fa18bb
## Use Cases
- **Social Media Moderation:** Automatically flag deepfake videos to prevent the spread of misinformation on platforms like Twitter, Instagram, or YouTube.
- **Video Authentication in Journalism:** Assist news agencies in verifying the authenticity of user-submitted or viral video content before publication.
- **Content Verification in Government & Security Agencies:** Help authorities validate the integrity of video evidence used in surveillance, intelligence, or public investigations.
## Future Improvements
- **Real-time Detection:** Integrate real-time video stream processing for on-the-fly deepfake detection.
- **Multimodal Analysis:** Combine audio and visual cues to improve accuracy and robustness of detection.
- **Model Optimization:** Explore lightweight architectures for faster inference and deployment on edge devices.
