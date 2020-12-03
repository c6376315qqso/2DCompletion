import loader
import torchvision
import torch
import solver
import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TRAIN = 1

if TRAIN == 1:
    loader.NEED_CREATE = 1
    solver.TRAIN = 1
else:
    loader.NEED_CREATE = 0
    solver.TRAIN = 0

    
dataset = loader.dataset()
num_of_data = len(dataset)
print(num_of_data)
train_data, test_data = torch.utils.data.random_split(
    dataset=dataset, lengths=[int(num_of_data * 0.8), num_of_data - int(num_of_data * 0.8)])

train_dataloader = torch.utils.data.DataLoader(train_data, 4, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_data, 4, shuffle=True, num_workers=0)



solve = solver.Solver(train_dataloader, test_dataloader)
solve.build_model()
solve.train()
loader.del_and_noise('timg.jpg', 1, '.')
image = cv2.imread('1.jpg')
image = cv2.resize(image, (512, 512))
recovered = solve.predict_single(image)

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(recovered)
plt.show()
recovered = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imwrite('recovered_1.jpg', recovered)
cv2.imwrite('image_1.jpg', image)