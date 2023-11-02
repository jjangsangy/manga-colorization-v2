import cv2
import os

from urllib.request import urlretrieve


def get_dnn_superres(upscale_factor: int = 3):
    """
    Returns a DNN Super Resolution model for the given upscale factor.

    Parameters
    ----------
    upscale_factor: int
        Upscale factor for the model (only 2x, 3x, and 4x are supported)

    Returns
    -------
    cv2.dnn_superres.DnnSuperResImpl
        Super Resolution Model
    """
    if upscale_factor not in range(2, 5):
        raise ValueError(
            f"Upscale factor must be between 2 and 4, got {upscale_factor}"
        )

    # Download the pretrained model
    filename = f"ESPCN_x{upscale_factor}.pb"
    if not os.path.exists(f"/tmp/{filename}"):
        urlretrieve(
            f"https://github.com/fannymonori/TF-ESPCN/raw/master/export/{filename}",
            f"/tmp/{filename}",
        )

    sr = cv2.dnn_superres.DnnSuperResImpl().create()
    sr.readModel(f"/tmp/{filename}")
    sr.setModel("espcn", upscale_factor)

    return sr


def upscale_single_image(image_path):
    img = cv2.imread(image_path)
    result = SR.upsample(img)
    cv2.imwrite(image_path, result)


SR = get_dnn_superres(upscale_factor=2)
