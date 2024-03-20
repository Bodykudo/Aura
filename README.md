# Aura

Aura is a comprehensive image processing toolkit designed to provide a wide range of image manipulation capabilities. It offers an intuitive interface and robust performance, making it a powerful tool for anyone interested in image processing, from hobbyists to professionals.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Contributors](#contributors)

## Features

Aura comes with a wide range of features:

- **Noise Addition**: Aura allows users to add various types of noise to images, including salt & pepper noise, uniform noise, and Gaussian noise.

- **Image Filtering**: Aura supports several image filtering techniques, including average filtering, Gaussian filtering, and median filtering.

- **Edge Detection**: Aura can apply various edge detection algorithms to images, including Sobel, Roberts, Prewitt, and Canny.

- **Frequency Domain Filters**: Aura supports applying frequency domain filters to images, including low pass and high pass filters. It can also create hybrid images using frequency domain.

- **Image Processing**: Aura can apply grayscale, normalization, and equalization to images, and it provides a histogram for visualizing the image's color distribution.

- **Thresholding**: Aura supports both global and local thresholding, allowing users to adjust the intensity levels in an image to separate objects from the background.

These features make Aura a versatile tool for a wide range of image processing tasks.

## Getting Started

### Prerequisites

- Node.js
- npm
- Python 3.6 or higher

### Installation

1. Clone this repository to your local machine.


2. Install the required front-end dependencies.

```bash
cd client
npm install
```

3. Run the front-end.
```bash
npm run dev
```

3. Open new trminal, and install the required server dependencies.
```bash
cd server
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the back-end.
```bash
uvicorn main:app --reload
```

5. Enjoy working with Aura :)

## Contributors

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Bodykudo" target="_black">
    <img src="https://avatars.githubusercontent.com/u/17731926?v=4" width="150px;" alt="Abdallah Magdy"/>
    <br />
    <sub><b>Abdallah Magdy</b></sub></a>
    <td align="center">
    <a href="https://github.com/Habiba-Mohsen" target="_black">
    <img src="https://avatars.githubusercontent.com/u/101303283?v=4" width="150px;" alt="Habiba Mohsen"/>
    <br />
    <sub><b>Habiba Mohsen</b></sub></a>
    </td>
    </td>
    <td align="center">
    <a href="https://github.com/Hazem-Raafat" target="_black">
    <img src="https://avatars.githubusercontent.com/u/100636693?v=4" width="150px;" alt="Hazem Raafat"/>
    <br />
    <sub><b>Hazem Raafat</b></sub></a>
    </td>
    <td align="center">
   <td align="">
    <a href="https://github.com/merna-abdelmoez" target="_black">
    <img src="https://avatars.githubusercontent.com/u/115110339?v=4" width="150px;" alt="Merna Abdelmoez"/>
    <br />
    <sub><b>Merna Abdelmoez</b></sub></a>
    </td>
   <td align="">
    <a href="https://github.com/raghdaneiazyy6" target="_black">
    <img src="https://avatars.githubusercontent.com/u/96526181?v=4" width="150px;" alt="Raghda Tarek"/>
    <br />
    <sub><b>Raghda Tarek</b></sub></a>
    </td>
    </tr>
 </table>

---
