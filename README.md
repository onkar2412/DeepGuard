# DeepFake Detection System

> A deep learning-based system for detecting manipulated (deepfake) videos using TensorFlow, OpenCV, and computer vision techniques.

---

## Description / Overview

DeepFake Detection System is an AI-powered application designed to identify manipulated videos generated using deepfake technology. The project utilizes deep learning and computer vision techniques to distinguish between authentic and synthetic videos with high accuracy.

This project demonstrates practical applications of artificial intelligence, image processing, and video analysis to address the growing challenge of detecting digitally manipulated media.

---

## Features

- Deepfake video classification
- High-accuracy deep learning model
- Video preprocessing pipeline
- Frame extraction and analysis
- Efficient model training and evaluation
- User-friendly prediction workflow
- Scalable architecture for future improvements

---

## How It Works

1. Loads the input video.
2. Extracts video frames.
3. Preprocesses the extracted frames.
4. Feeds frames into the trained TensorFlow model.
5. Predicts whether the video is **Real** or **DeepFake**.
6. Displays the prediction results.

---

## System Architecture

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
Image Preprocessing
     │
     ▼
Feature Extraction
     │
     ▼
Deep Learning Model
     │
     ▼
Prediction
     │
     ▼
Result (Real / Fake)
```

---

## Dataset

This project is trained using the **Celeb-DF Dataset**, one of the most widely used datasets for deepfake detection research.

The dataset contains:

- Real videos
- DeepFake videos
- High-quality facial manipulation samples

---

## Technologies Used

### Programming Language

- Python

### Frameworks & Libraries

- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Tools

- Visual Studio Code
- Git
- GitHub

---

## Project Structure

```text
DeepFake-Detection-System/
│
├── dataset/
├── models/
├── notebooks/
├── screenshots/
├── src/
├── outputs/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/onkar2412/DeepFake-Detection-System.git
```

### Navigate to the Project Directory

```bash
cd DeepFake-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application using:

```bash
python main.py
```

Follow the prompts to load a video and perform deepfake detection.

---

## Requirements

- Python 3.9 or later
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Results

The system successfully:

- Detects manipulated videos using deep learning.
- Classifies videos as **Real** or **DeepFake**.
- Processes videos efficiently with optimized preprocessing.
- Provides reliable predictions for research and educational purposes.

---

## Future Enhancements

- Real-time webcam deepfake detection
- Improved CNN and Transformer architectures
- Web-based interface using Flask or FastAPI
- Mobile application support
- Cloud deployment
- Explainable AI (XAI) visualizations
- Support for additional benchmark datasets

---

## Contributing

Contributions are welcome.

1. Fork this repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.

For complete details, please refer to the [LICENSE](LICENSE) file.

---

## Acknowledgments

- TensorFlow
- OpenCV
- Celeb-DF Dataset
- Python Community

---

## Author

**Onkar Shrikonda**

AI & Data Science Engineer | Python Developer | Computer Vision Enthusiast

---

## Contact

**GitHub:** https://github.com/onkar2412

**LinkedIn:** https://www.linkedin.com/in/onkar-shrikonda/

**Email:** shrikonda.onkar@gmail.com
