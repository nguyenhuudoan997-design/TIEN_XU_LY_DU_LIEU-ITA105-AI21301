import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_images(images, titles, main_title, cols=5):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(10, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# BÀI 1: CĂN HỘ / MẶT TIỀN
print("Đang xử lý Bài 1...")
img1 = cv2.imread('apartment.jpg')
if img1 is not None:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_resized = cv2.resize(img1, (224, 224))
    
    b1_results = []
    b1_titles = []
    for i in range(5):
    
        M = cv2.getRotationMatrix2D((112, 112), np.random.uniform(-15, 15), 1.0)
        aug = cv2.warpAffine(img1_resized, M, (224, 224))
        if np.random.rand() > 0.5: aug = cv2.flip(aug, 1)
        br = np.random.uniform(0.8, 1.2)
        aug = np.clip(aug * br, 0, 255).astype(np.uint8)
        
        gray = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY) / 255.0
        b1_results.append(gray)
        b1_titles.append(f"Augmented {i+1}")
    
    display_images([img1_resized/255.0]*5 + b1_results, ["Gốc"]*5 + b1_titles, "Bài 1: Căn hộ")

# BÀI 2: XE Ô TÔ / XE MÁY
print("Đang xử lý Bài 2...")
img2 = cv2.imread('vehicle.jpg')
if img2 is not None:
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_resized = cv2.resize(img2, (224, 224))
    
    M = cv2.getRotationMatrix2D((112, 112), np.random.uniform(-10, 10), 1.0)
    aug2 = cv2.warpAffine(img2_resized, M, (224, 224))
    br2 = np.random.uniform(0.85, 1.15)
    aug2 = np.clip(aug2 * br2, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 15, aug2.shape).astype(np.int16)
    aug2 = np.clip(aug2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    display_images([img2_resized/255.0, aug2/255.0], ["Gốc", "Augmented + Noise"], "Bài 2: Xe cộ")

# BÀI 3: TRÁI CÂY / NÔNG SẢN 
print("Đang xử lý Bài 3...")
img3 = cv2.imread('fruit.jpg')
if img3 is not None:
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3_resized = cv2.resize(img3, (224, 224))
    
    b3_results = []
    for i in range(9):
       
        M = cv2.getRotationMatrix2D((112, 112), np.random.uniform(-30, 30), np.random.uniform(0.7, 1.2))
        aug = cv2.warpAffine(img3_resized, M, (224, 224))
        if np.random.rand() > 0.5: aug = cv2.flip(aug, 1)
        b3_results.append(aug / 255.0)
    
    display_images(b3_results, [f"Aug {i+1}" for i in range(9)], "Bài 3: Trái cây", cols=3)

# BÀI 4: PHÒNG / NỘI THẤT 
print("Đang xử lý Bài 4...")
img4 = cv2.imread('interior.jpg')
if img4 is not None:
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img4_resized = cv2.resize(img4, (224, 224))
    
    b4_results = []
    for i in range(3):
 
        M = cv2.getRotationMatrix2D((112, 112), np.random.uniform(-15, 15), 1.0)
        aug = cv2.warpAffine(img4_resized, M, (224, 224))
        aug = cv2.flip(aug, 1)
        aug = np.clip(aug * 1.2, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY) / 255.0
        b4_results.append(gray)
    
    display_images([img4_resized/255.0] + b4_results, ["Gốc", "Aug 1", "Aug 2", "Aug 3"], "Bài 4: Nội thất")
