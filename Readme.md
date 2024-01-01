# Nepali Sign Language Recognition System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kanchansapkota27/Nepali-Sign-Language-Recognition/)
![Stars](https://img.shields.io/github/stars/kanchansapkota27/Nepali-Sign-Language-Recognition?style=social)
![Windows](https://svgshare.com/i/ZhY.svg)
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)
![ProjectType](https://img.shields.io/badge/ProjectType-University-blue)
[![Paper Staus](https://img.shields.io/badge/PaperStatus-Published-green)](https://www.appleacademicpress.com/machine-learning-algorithms-in-security-analytics-applications-principles-and-practices/9781774912393)

# Abstract

It has always been difficult to communicate and interact with those people
who are unable to speak or listen. Human translators somewhat try to
bridge the communication gap between the deaf-mute community and those
who do not know how to read and use the sign language. However, they
are limited in number and are not available everywhere, all the time. So,
to solve this problem, we can use various computer science technologies
to detect and classify the sign language gestures. This chapter proposes a
system to detect and recognize dynamic Nepali Sign Language (NSL) in real
time using a deep learning technique with the help of computer vision. The
proposed approach takes video input from the user, extracts its frames, and
classifies the sequence of images using a combined model of Convolutional
Neural Network (CNN) and Long Short-Term Memory (LSTM). We have
used InceptionV3, a transfer learning approach to extract spatial features
and LSTM, a type of Recurrent Neural Network (RNN) to recognize the
temporal features. The dataset is collected manually by capturing videos
using a smartphone for five different classes.

# Sample Screenshots

![Splash](/assets/splash.png)
![Options](/assets/other_options.png)
![Live Mode](/assets/livemode.png)
![Recording Mode](/assets/recording.png)
![Completed](/assets/completed.png)


# Getting Started

```python
# Clone the repo
git clone https://github.com/kanchansapkota27/Nepali-Sign-Language-Recognition.git
```

```python
# Navigate to project directory
cd Nepali-Sign-Language-Recognition
```
```python
#Create a virual environment and activate it
python -m venv
```

```python
# Activate the envrionment
venv/Scripts/activate.bat
```

```python
# Install the requirements
pip install -r requirements.txt
```

```python
# Run main GUI
python app.py 
```

## Authors

- [Kanchan Sapkota](https://github.com/kanchansapkota27)
- [Sailesh Rana](https://www.github.com/Sailesh01)
