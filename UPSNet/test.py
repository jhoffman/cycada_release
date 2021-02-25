from upsnet.dataset.json_dataset import JsonDataset

a = JsonDataset('cityscapes', 'data/cityscapes', 'data/cityscapes/annotations/instancesonly_gtFine_val.json')
a.get_roidb(gt=True)