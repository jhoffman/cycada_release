class Rotater(object):

    def __init__(self, dataset, orientations=6, transform=None,
                 target_transform=None):
        self.dataset = dataset
        self.orientations = orientations
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        im, target = self.dataset[index]
        rotation = index % self.orientations
        degrees = 360 / self.orientations * rotation
        im = im.rotate(degrees)
        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target, degrees

    def __len__(self):
        return len(self.dataset)
