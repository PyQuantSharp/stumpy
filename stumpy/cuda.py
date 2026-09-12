import shutil
import subprocess


def _nvidia_smi_is_available():
    """
    Checks whether `nvidia-smi` is available

    Parameters
    ----------
    None

    Returns
    -------
    bool
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def is_available():
    """
    Checks whether `cuda` is available

    Parameters
    ----------
    None

    Returns
    -------
    bool
    """
    if not _nvidia_smi_is_available():
        return False
    # if not _other_condition():
    #     return False

    return True
