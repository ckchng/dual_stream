from torchmetrics import JaccardIndex


def get_seg_metrics(config, task=None, reduction='none'):
    # Switch to binary IoU when there is only one class output channel.
    if task is None:
        task = 'binary' if config.num_class == 1 else 'multiclass'

    if task == 'binary':
        metrics = JaccardIndex(task='binary', threshold=config.pred_threshold)
    else:
        metrics = JaccardIndex(task='multiclass', num_classes=config.num_class,
                               ignore_index=config.ignore_index, average=reduction)
    return metrics