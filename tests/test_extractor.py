"""
Unit tests for O*NET Extractor
"""

from onet_extractor import ONetExtractor
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(
    0, str(Path(__file__).parent.parent / 'src' / 'backend' / 'etl'))


class TestONetExtractor:
    """Test cases for ONetExtractor class"""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create an ONetExtractor instance with temporary directory"""
        return ONetExtractor(data_dir=str(tmp_path))

    def test_initialization(self, extractor, tmp_path):
        """Test that extractor initializes with correct paths"""
        assert extractor.data_dir == tmp_path
        assert extractor.raw_dir.exists()
        assert extractor.key_files is not None
        assert len(extractor.key_files) > 0

    def test_data_directory_creation(self, extractor):
        """Test that necessary directories are created"""
        assert extractor.raw_dir.exists()
        assert extractor.raw_dir.is_dir()

    def test_key_files_defined(self, extractor):
        """Test that key O*NET files are defined"""
        expected_keys = ['knowledge', 'skills', 'abilities', 'occupation_data']
        for key in expected_keys:
            assert key in extractor.key_files

    @patch('onet_extractor.requests.get')
    def test_download_success(self, mock_get, extractor):
        """Test successful download of O*NET database"""
        # Mock successful download
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'test data'])
        mock_response.headers = {'content-length': '1000'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = extractor.download_onet_database(force_download=True)
        assert result is True

    def test_clean_dataframe(self, extractor):
        """Test DataFrame cleaning functionality"""
        df = pd.DataFrame({
            'col1': ['  value1  ', 'value2', ''],
            'col2': ['value3', '  value4  ', None]
        })

        cleaned = extractor._clean_dataframe(df)

        # Check whitespace stripping
        assert cleaned.loc[0, 'col1'] == 'value1'
        assert cleaned.loc[1, 'col2'] == 'value4'
