from dataset.volleyball import VolleyBallDataset
from dataset.videocoatt import VideoCoAttDataset
from dataset.videoattentiontarget import VideoAttentionTargetDataset
# from dataset.toy import ToyDataset

def dataset_generator(cfg, mode):
    print(f'{cfg.data.name} dataset')
    if cfg.data.name == 'volleyball':
        data_set = VolleyBallDataset(cfg, mode)
    elif cfg.data.name == 'videocoatt':
        if mode == 'valid':
            mode = 'validate'
        data_set = VideoCoAttDataset(cfg, mode)
    elif cfg.data.name == 'videoattentiontarget':
        if mode == 'valid':
            mode = 'test'
        data_set = VideoAttentionTargetDataset(cfg, mode)
    elif cfg.data.name == 'toy_dataset':
        pass
    else:
        assert True, 'cfg.data.name is incorrect'

    return data_set