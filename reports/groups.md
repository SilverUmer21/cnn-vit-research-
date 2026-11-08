# Group assignment

Images: 54,304. Edges: byte-identical files, plus same-class near-duplicate pairs with grayscale RMS <= T (from `src/dataset/duplicates.py`). Cross-class pairs are never merged. T = 0 means byte-identical files only.

## 1. Group structure by threshold

| T (RMS) | Groups | Groups with 2+ images | Images in such groups | Largest group | Five largest | Groups spanning train and val |
|---|---|---|---|---|---|---|
| 0 | 54,283 | 21 | 42 | 2 | 2, 2, 2, 2, 2 | 10 |
| 2 | 54,282 | 22 | 44 | 2 | 2, 2, 2, 2, 2 | 11 |
| 3 | 54,277 | 27 | 54 | 2 | 2, 2, 2, 2, 2 | 12 |
| 4 | 54,275 | 29 | 58 | 2 | 2, 2, 2, 2, 2 | 12 |
| 5 | 54,271 | 33 | 66 | 2 | 2, 2, 2, 2, 2 | 13 |
| 6 | 54,200 | 73 | 177 | 10 | 10, 7, 7, 6, 4 | 29 |
| 7 | 53,993 | 100 | 411 | 182 | 182, 14, 8, 4, 4 | 37 |
| 8 | 53,691 | 144 | 757 | 434 | 434, 11, 9, 7, 6 | 42 |
| 10 | 52,716 | 316 | 1,904 | 573 | 573, 426, 206, 13, 11 | 101 |
| 12 | 50,349 | 573 | 4,528 | 1170 | 1170, 587, 392, 379, 221 | 203 |

## 2. Chosen threshold: T = 6

Groups: 54,200. Groups with 2+ images: 73 covering 177 images. Largest group: 10.

| Group size | Groups |
|---|---|
| 1 | 54,127 |
| 2 | 62 |
| 3 | 5 |
| 4 | 2 |
| 6 | 1 |
| 7 | 2 |
| 10 | 1 |

Classes with the largest groups (a group must fit inside one split; test quota = 10% of the class):

| Class | Class size | Largest group | Test quota |
|---|---|---|---|
| Tomato___healthy | 1,591 | 10 | 159 |
| Apple___Apple_scab | 630 | 2 | 63 |
| Cherry_(including_sour)___Powdery_mildew | 1,052 | 2 | 105 |
| Pepper,_bell___healthy | 1,477 | 2 | 148 |
| Corn_(maize)___Northern_Leaf_Blight | 985 | 2 | 98 |
| Apple___healthy | 1,645 | 2 | 164 |
| Tomato___Late_blight | 1,909 | 2 | 191 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 | 2 | 536 |

Groups that span the shipped train and val: 29 (84 images). These were leaking in the shipped split.
