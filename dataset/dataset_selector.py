from dataset.volleyball import VolleyBallDataset
# from dataset.videocoatt import VideoCoAttDataset
# from dataset.videoattentiontarget import VideoAttentionTargetDataset
# from dataset.toy import ToyDataset

def dataset_generator(cfg, mode):

    if cfg.data.name == 'volleyball':
        data_set = VolleyBallDataset(cfg, mode)
    elif cfg.data.name == 'videocoatt':
        pass
    elif cfg.data.name == 'videoattentiontarget':
        pass
    elif cfg.data.name == 'toy_dataset':
        pass
    else:
        assert True, 'cfg.data.name is incorrect'

    return data_set