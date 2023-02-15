# encoding: utf-8


from .roberta_ists import RobertaISTS


def build_model(cfg):
    model = RobertaISTS(cfg.MODEL.NUM_CLASSES, cfg.MODEL.DROPOUT, cfg.MODEL.HIDDEN_NEURONS)
    return model
