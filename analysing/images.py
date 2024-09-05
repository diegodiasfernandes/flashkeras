from utils.kerasimports import *
from utils.otherimports import *

def show_images_from_batch(batches: DirectoryIterator, num_images: int = 1, fig_size: tuple[int, int] = (15,5)):
    all_images = []
    images, labels = (None, None)
    while True:
        images, labels = next(batches)
        for img in images:
            all_images.append(img)
        if batches.batch_index == 0: break
    
    num_images = min(num_images, len(all_images))
    
    plt.figure(figsize=fig_size)
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(all_images[i])
        plt.axis('off')
    
    plt.show()

def show_images_nparray(images: np.ndarray, num_images: int = 1, fig_size: tuple[int, int] = (15,5)):
    num_images = min(num_images, len(images))

    plt.figure(figsize=fig_size)
    
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')  # Remove os eixos

    plt.show()

def show_images_from_directory(dir_path: str, num_images=1):

    files = os.listdir(dir_path)
    
    valid_extentions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extentions]
    
    num_images = min(num_images, len(images))
    
    plt.figure(figsize=(15, 5))
    
    for i in range(num_images):
        img_path = os.path.join(dir_path, images[i])
        img = mpimg.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.show()