from .nms.gpu_nms import gpu_nms
from .nms.cpu_nms import cpu_nms, cpu_soft_nms



# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if force_cpu:
        # return cpu_soft_nms(dets, thresh, method = 0)
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
