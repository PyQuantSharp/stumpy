import subprocess
from unittest.mock import MagicMock, patch

from stumpy import cuda


@patch("stumpy.cuda.shutil.which")
def test_nvidia_smi_not_in_path(mock_which):
    """Test when nvidia-smi executable is not found in the system PATH."""
    mock_which.return_value = None

    assert cuda._nvidia_smi_is_available() is False
    mock_which.assert_called_once_with("nvidia-smi")


@patch("stumpy.cuda.subprocess.run")
@patch("stumpy.cuda.shutil.which")
def test_nvidia_smi_success(mock_which, mock_run):
    """Test when nvidia-smi is found and executes successfully."""
    mock_which.return_value = "/usr/bin/nvidia-smi"

    # Mock a successful subprocess execution (returncode 0)
    mock_response = MagicMock()
    mock_response.returncode = 0
    mock_run.return_value = mock_response

    assert cuda._nvidia_smi_is_available() is True
    mock_run.assert_called_once_with(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@patch("stumpy.cuda.subprocess.run")
@patch("stumpy.cuda.shutil.which")
def test_nvidia_smi_error_return_code(mock_which, mock_run):
    """Test when nvidia-smi exists but returns an error code (e.g., driver issues)."""
    mock_which.return_value = "/usr/bin/nvidia-smi"

    # Mock a failing subprocess execution (returncode non-zero)
    mock_response = MagicMock()
    mock_response.returncode = 1
    mock_run.return_value = mock_response

    assert cuda._nvidia_smi_is_available() is False


@patch("stumpy.cuda.subprocess.run")
@patch("stumpy.cuda.shutil.which")
def test_nvidia_smi_exception(mock_which, mock_run):
    """Test when subprocess.run throws an unexpected exception."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    mock_run.side_effect = RuntimeError("Unexpected system error")

    assert cuda._nvidia_smi_is_available() is False


def test_cuda_is_available():
    with patch.object(cuda, "_nvidia_smi_is_available", return_value=True):
        assert cuda.is_available() is True


def test_not_cuda_is_available():
    with patch.object(cuda, "_nvidia_smi_is_available", return_value=False):
        assert cuda.is_available() is False
