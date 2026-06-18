# ✋ Giving Voice Through Hands — ASL Recognition Project

A real-time sign language recognition project built using machine learning and hand landmark detection.

This system recognizes hand gestures, converts them into letters, builds complete words, and can speak the generated output aloud.

Created with the idea of making communication feel more accessible through computer vision and interactive feedback.

Approximately **90,000 images** were used during training and experimentation.

---

## ✨ Features

- Real-time hand detection
- Sign recognition
- Letter-by-letter prediction
- Word generation
- Text-to-speech output
- Landmark extraction pipeline
- Dataset-based training
- Interactive keyboard controls

---

## 🎮 Controls

| Key | Action |
|------|--------|
| Spacebar | Speak the completed word |
| C | Clear current word |
| Q | Exit application |

---

## 📸 Screenshots

### No Hand Detected

System waits until a hand appears.

![No Hand](images/Screenshot%20(220).png)

---

### Recognizing Letter B

![B](images/Screenshot%20(222).png)

---

### Recognizing Letter M

![M](images/Screenshot%20(223).png)

---

### Recognizing Letter W

![W](images/Screenshot%20(225).png)

---

### Final Output — BMW + Speech

After completing the word, press **Spacebar** to speak the result.

![BMW](images/Screenshot%20(227).png)

---

## 🧠 Training Pipeline

1. Collect images
2. Extract hand landmarks
3. Generate datasets
4. Train model
5. Run live prediction

Training scale:

```plaintext
~90,000 images
```

---

## 📂 Project Structure

```plaintext
ASL-project/
├── data/
├── main.py
├── train_model.py
├── extract_landmarkers.py
├── requirements.txt
├── README.md
```

---

## 🚀 Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start application:

```bash
python main.py
```

---

## 📦 Model Access

The trained model (`asl_model.pkl`) is not included in this repository.

This project requires the trained model file to run full real-time prediction and speech output.

If you would like access to the model for testing, learning, or running the complete project, please contact:

📧 moazzinhussain8@gmail.com

Please include:
- Your name
- Purpose of use
- Project or learning context

---

## ⚠ Notes

Large trained model files are not included in this repository.

---

## 🛠 Built With

- Python
- Machine Learning
- Computer Vision
- CSV Datasets
- Hand Landmark Detection
- Text To Speech