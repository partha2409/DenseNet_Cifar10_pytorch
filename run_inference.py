import torch
from model import DenseNet
from data_generator import TestDataGenerator
import pickle
import utils
from torch.utils.data import DataLoader


def run_inference(model_dir=None, model_path=None, data_path=None):
    hp_file = model_dir + 'hp.pkl'
    f = open(hp_file, "rb")
    hp = pickle.load(f)

    model = DenseNet().to(device)
    ckpt = torch.load(model_dir+model_path)
    model.load_state_dict(ckpt['net'])

    dataset = TestDataGenerator(data_path)
    iterator = DataLoader(dataset=dataset, batch_size=50, num_workers=hp['num_workers'], pin_memory=True,
                          shuffle=False, drop_last=True)

    with torch.no_grad():
        acc = 0
        for i, (img, lab) in enumerate(iterator):
            img = img.to(device, dtype=torch.float)
            lab = lab.to(device, dtype=torch.float)
            logits = model(img)
            # acc_per_batch
            logits = logits.detach().cpu().numpy()
            label = lab.detach().cpu().numpy()
            acc += utils.classification_accuracy(logits, label)

        accuracy = acc/(i+1)
        print("Model test accuracy = {}".format(accuracy))


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir = ""   # add checkpoint root dir eg: "checkpoints/DenseNet_20201226_022024/"
    model_path = ""  # model name eg: "densenet_cifar10_epoch99.pth"
    test_data_path = "cifar-10-batches-py/"

    run_inference(model_dir, model_path, test_data_path)